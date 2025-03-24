/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "jni_layer_constants.h"

#include <executorch/extension/android/jni/log.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <tokenizers_cpp.h>

#ifdef ET_USE_THREADPOOL
#include <cpuinfo.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

using namespace executorch::extension;
using namespace torch::executor;

namespace executorch::extension {
class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  explicit TensorHybrid(exec_aten::Tensor tensor) {}

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromTensor(const exec_aten::Tensor& tensor) {
    // Java wrapper currently only supports contiguous tensors.

    const auto scalarType = tensor.scalar_type();

    if (scalar_type_to_java_dtype.count(scalarType) == 0) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "exec_aten::Tensor scalar type %d is not supported on java side",
          scalarType);
    }
    int jdtype = scalar_type_to_java_dtype.at(scalarType);

    const auto& tensor_shape = tensor.sizes();
    std::vector<jlong> tensor_shape_vec;
    for (const auto& s : tensor_shape) {
      tensor_shape_vec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensor_shape_vec.size());
    jTensorShape->setRegion(
        0, tensor_shape_vec.size(), tensor_shape_vec.data());

    static auto cls = TensorHybrid::javaClassStatic();
    // Note: this is safe as long as the data stored in tensor is valid; the
    // data won't go out of scope as long as the Method for the inference is
    // valid and there is no other inference call. Java layer picks up this
    // value immediately so the data is valid.
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
        facebook::jni::JByteBuffer::wrapBytes(
            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
    jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

    static const auto jMethodNewTensor =
        cls->getStaticMethod<facebook::jni::local_ref<TensorHybrid::javaobject>(
            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
            facebook::jni::alias_ref<jlongArray>,
            jint,
            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
    return jMethodNewTensor(
        cls, jTensorBuffer, jTensorShape, jdtype, makeCxxInstance(tensor));
  }

 private:
  friend HybridBase;
};

class JEValue : public facebook::jni::JavaClass<JEValue> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/EValue;";

  constexpr static int kTypeCodeTensor = 1;
  constexpr static int kTypeCodeString = 2;
  constexpr static int kTypeCodeDouble = 3;
  constexpr static int kTypeCodeInt = 4;
  constexpr static int kTypeCodeBool = 5;

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(EValue evalue) {
    if (evalue.isTensor()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
      return jMethodTensor(
          JEValue::javaClassStatic(),
          TensorHybrid::newJTensorFromTensor(evalue.toTensor()));
    } else if (evalue.isInt()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jlong)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toInt());
    } else if (evalue.isDouble()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jdouble)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toDouble());
    } else if (evalue.isBool()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jboolean)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toBool());
    } else if (evalue.isString()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<jstring>)>("from");
      std::string str =
          std::string(evalue.toString().begin(), evalue.toString().end());
      return jMethodTensor(
          JEValue::javaClassStatic(), facebook::jni::make_jstring(str));
    }
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unsupported EValue type: %d",
        evalue.tag);
  }

  static TensorPtr JEValueToTensorImpl(
      facebook::jni::alias_ref<JEValue> JEValue) {
    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");
    const auto typeCode = JEValue->getFieldValue(typeCodeField);
    if (JEValue::kTypeCodeTensor == typeCode) {
      static const auto jMethodGetTensor =
          JEValue::javaClassStatic()
              ->getMethod<facebook::jni::alias_ref<TensorHybrid::javaobject>()>(
                  "toTensor");
      auto jtensor = jMethodGetTensor(JEValue);

      static auto cls = TensorHybrid::javaClassStatic();
      static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
      jint jdtype = dtypeMethod(jtensor);

      static const auto shapeField = cls->getField<jlongArray>("shape");
      auto jshape = jtensor->getFieldValue(shapeField);

      static auto dataBufferMethod = cls->getMethod<
          facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
          "getRawDataBuffer");
      facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
          dataBufferMethod(jtensor);

      const auto rank = jshape->size();

      const auto shapeArr = jshape->getRegion(0, rank);
      std::vector<exec_aten::SizesType> shape_vec;
      shape_vec.reserve(rank);

      auto numel = 1;
      for (int i = 0; i < rank; i++) {
        shape_vec.push_back(shapeArr[i]);
      }
      for (int i = rank - 1; i >= 0; --i) {
        numel *= shapeArr[i];
      }
      JNIEnv* jni = facebook::jni::Environment::current();
      if (java_dtype_to_scalar_type.count(jdtype) == 0) {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Unknown Tensor jdtype %d",
            jdtype);
      }
      ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
      const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
      if (dataCapacity != numel) {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Tensor dimensions(elements number:%d inconsistent with buffer capacity(%d)",
            numel,
            dataCapacity);
      }
      return from_blob(
          jni->GetDirectBufferAddress(jbuffer.get()), shape_vec, scalar_type);
    }
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unknown EValue typeCode %d",
        typeCode);
  }
};

class ExecuTorchJni : public facebook::jni::HybridClass<ExecuTorchJni> {
 private:
  friend HybridBase;
  std::unique_ptr<Module> module_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/executorch/NativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      jint loadMode) {
    return makeCxxInstance(modelPath, loadMode);
  }

  ExecuTorchJni(facebook::jni::alias_ref<jstring> modelPath, jint loadMode) {
    Module::LoadMode load_mode = Module::LoadMode::Mmap;
    if (loadMode == 0) {
      load_mode = Module::LoadMode::File;
    } else if (loadMode == 1) {
      load_mode = Module::LoadMode::Mmap;
    } else if (loadMode == 2) {
      load_mode = Module::LoadMode::MmapUseMlock;
    } else if (loadMode == 3) {
      load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors;
    }

    module_ = std::make_unique<Module>(modelPath->toStdString(), load_mode);

#ifdef ET_USE_THREADPOOL
    // Default to using cores/2 threadpool threads. The long-term plan is to
    // improve performant core detection in CPUInfo, but for now we can use
    // cores/2 as a sane default.
    //
    // Based on testing, this is almost universally faster than using all
    // cores, as efficiency cores can be quite slow. In extreme cases, using
    // all cores can be 10x slower than using cores/2.
    //
    // TODO Allow overriding this default from Java.
    auto threadpool = executorch::extension::threadpool::get_threadpool();
    if (threadpool) {
      int thread_count = cpuinfo_get_processors_count() / 2;
      if (thread_count > 0) {
        threadpool->_unsafe_reset_threadpool(thread_count);
      }
    }
#endif
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method("forward", jinputs);
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> execute(
      facebook::jni::alias_ref<jstring> methodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method(methodName->toStdString(), jinputs);
  }

  jint load_method(facebook::jni::alias_ref<jstring> methodName) {
    return static_cast<jint>(module_->load_method(methodName->toStdString()));
  }

  jint get_number_of_inputs() {
    const auto method_meta = module_->method_meta("forward");

    if (method_meta.ok()) {
      return static_cast<jint>(method_meta->num_inputs());
    }
  }

  jint get_input_type(jint index) {
    const auto method_meta = module_->method_meta("forward");

    if (method_meta.ok()) {
      const auto input_meta = method_meta->input_tensor_meta(index);
      if (input_meta.ok()) {
        switch (input_meta->scalar_type()) {
          case ScalarType::Byte:
            return static_cast<jint>(0);
          case ScalarType::Int:
            return static_cast<jint>(1);
          case ScalarType::Long:
            return static_cast<jint>(2);
          case ScalarType::Float:
            return static_cast<jint>(3);
          case ScalarType::Double:
            return static_cast<jint>(4);
          default:
            return static_cast<jint>(-1);
        }
      }
    }

    return static_cast<jint>(-1);
  }

  facebook::jni::local_ref<facebook::jni::JArrayLong> get_input_shape(
      jint index) {
    auto method_meta = module_->method_meta("forward");
    if (!method_meta.ok()) {
      return nullptr;
    }

    const auto input_meta = method_meta->input_tensor_meta(index);
    if (!input_meta.ok()) {
      return nullptr;
    }

    const auto shape = input_meta->sizes();
    std::vector<jlong> jlong_shape(shape.begin(), shape.end());

    auto shapeArray = facebook::jni::make_long_array(jlong_shape.size());
    shapeArray->setRegion(0, jlong_shape.size(), jlong_shape.data());
    return shapeArray;
  }

  jint get_number_of_outputs() {
    const auto method_meta = module_->method_meta("forward");

    if (method_meta.ok()) {
      return static_cast<jint>(method_meta->num_outputs());
    }
  }

  jint get_output_type(jint index) {
    const auto method_meta = module_->method_meta("forward");

    if (method_meta.ok()) {
      const auto output_meta = method_meta->output_tensor_meta(index);
      if (output_meta.ok()) {
        switch (output_meta->scalar_type()) {
          case ScalarType::Byte:
            return static_cast<jint>(0);
          case ScalarType::Int:
            return static_cast<jint>(1);
          case ScalarType::Long:
            return static_cast<jint>(2);
          case ScalarType::Float:
            return static_cast<jint>(3);
          case ScalarType::Double:
            return static_cast<jint>(4);
          default:
            return static_cast<jint>(-1);
        }
      }
    }

    return static_cast<jint>(-1);
  }

  facebook::jni::local_ref<facebook::jni::JArrayLong> get_output_shape(
      jint index) {
    auto method_meta = module_->method_meta("forward");
    if (!method_meta.ok()) {
      return nullptr;
    }

    const auto output_meta = method_meta->output_tensor_meta(index);
    if (!output_meta.ok()) {
      return nullptr;
    }

    const auto shape = output_meta->sizes();
    std::vector<jlong> jlong_shape(shape.begin(), shape.end());

    auto shapeArray = facebook::jni::make_long_array(jlong_shape.size());
    shapeArray->setRegion(0, jlong_shape.size(), jlong_shape.data());
    return shapeArray;
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> execute_method(
      std::string method,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    // If no inputs is given, it will run with sample inputs (ones)
    if (jinputs->size() == 0) {
      if (module_->load_method(method) != Error::Ok) {
        return {};
      }
      auto&& underlying_method = module_->methods_[method].method;
      auto&& buf = prepare_input_tensors(*underlying_method);
      auto result = underlying_method->execute();
      if (result != Error::Ok) {
        return {};
      }
      facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> jresult =
          facebook::jni::JArrayClass<JEValue>::newArray(
              underlying_method->outputs_size());

      for (int i = 0; i < underlying_method->outputs_size(); i++) {
        auto jevalue =
            JEValue::newJEValueFromEValue(underlying_method->get_output(i));
        jresult->setElement(i, *jevalue);
      }
      return jresult;
    }

    std::vector<EValue> evalues;
    std::vector<TensorPtr> tensors;

    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");

    for (int i = 0; i < jinputs->size(); i++) {
      auto jevalue = jinputs->getElement(i);
      const auto typeCode = jevalue->getFieldValue(typeCodeField);
      if (typeCode == JEValue::kTypeCodeTensor) {
        tensors.emplace_back(JEValue::JEValueToTensorImpl(jevalue));
        evalues.emplace_back(tensors.back());
      } else if (typeCode == JEValue::kTypeCodeInt) {
        int64_t value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      } else if (typeCode == JEValue::kTypeCodeDouble) {
        double value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      } else if (typeCode == JEValue::kTypeCodeBool) {
        bool value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      }
    }

#ifdef EXECUTORCH_ANDROID_PROFILING
    auto start = std::chrono::high_resolution_clock::now();
    auto result = module_->execute(method, evalues);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    ET_LOG(Debug, "Execution time: %lld ms.", duration);

#else
    auto result = module_->execute(method, evalues);

#endif

    if (!result.ok()) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Execution of method %s failed with status 0x%" PRIx32,
          method.c_str(),
          static_cast<error_code_t>(result.error()));
      return {};
    }

    facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> jresult =
        facebook::jni::JArrayClass<JEValue>::newArray(result.get().size());

    for (int i = 0; i < result.get().size(); i++) {
      auto jevalue = JEValue::newJEValueFromEValue(result.get()[i]);
      jresult->setElement(i, *jevalue);
    }

    return jresult;
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  readLogBuffer() {
#ifdef __ANDROID__

    facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> ret;

    access_log_buffer([&](std::vector<log_entry>& buffer) {
      const auto size = buffer.size();
      ret = facebook::jni::JArrayClass<jstring>::newArray(size);
      for (auto i = 0u; i < size; i++) {
        const auto& entry = buffer[i];
        // Format the log entry as "[TIMESTAMP FUNCTION FILE:LINE] LEVEL
        // MESSAGE".
        std::stringstream ss;
        ss << "[" << entry.timestamp << " " << entry.function << " "
           << entry.filename << ":" << entry.line << "] "
           << static_cast<char>(entry.level) << " " << entry.message;

        facebook::jni::local_ref<facebook::jni::JString> jstr_message =
            facebook::jni::make_jstring(ss.str().c_str());
        (*ret)[i] = jstr_message;
      }
    });

    return ret;
#else
    return facebook::jni::JArrayClass<String>::newArray(0);
#endif
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
        makeNativeMethod("forward", ExecuTorchJni::forward),
        makeNativeMethod("execute", ExecuTorchJni::execute),
        makeNativeMethod("loadMethod", ExecuTorchJni::load_method),
        makeNativeMethod("readLogBuffer", ExecuTorchJni::readLogBuffer),
        makeNativeMethod(
            "getNumberOfInputs", ExecuTorchJni::get_number_of_inputs),
        makeNativeMethod("getInputType", ExecuTorchJni::get_input_type),
        makeNativeMethod("getInputShape", ExecuTorchJni::get_input_shape),
        makeNativeMethod(
            "getNumberOfOutputs", ExecuTorchJni::get_number_of_inputs),
        makeNativeMethod("getOutputType", ExecuTorchJni::get_output_type),
        makeNativeMethod("getOutputShape", ExecuTorchJni::get_output_shape),
    });
  }
};

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

class ExecuTorchHuggingFaceTokenizerJni
    : public facebook::jni::HybridClass<ExecuTorchHuggingFaceTokenizerJni> {
 private:
  friend HybridBase;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/HuggingFaceTokenizer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> jsonPath) {
    return makeCxxInstance(jsonPath);
  }

  ExecuTorchHuggingFaceTokenizerJni(
      facebook::jni::alias_ref<jstring> jsonPath) {
    auto blob = LoadBytesFromFile(jsonPath->toStdString());
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  }

  facebook::jni::local_ref<jlongArray> encode(
      facebook::jni::alias_ref<jstring> text) {
    std::vector<int32_t> encoded = tokenizer_->Encode(text->toStdString());
    facebook::jni::local_ref<jlongArray> result =
        facebook::jni::make_long_array(static_cast<jsize>(encoded.size()));
    std::vector<jlong> encoded_long(encoded.begin(), encoded.end());
    result->setRegion(0, encoded_long.size(), encoded_long.data());

    return result;
  }

  facebook::jni::local_ref<jstring> decode(
      facebook::jni::alias_ref<jlongArray> tokenIds) {
    std::vector<jlong> token_ids_jlong(tokenIds->size());
    std::vector<int32_t> token_ids(tokenIds->size());
    tokenIds->getRegion(0, tokenIds->size(), token_ids_jlong.data());
    for (int i = 0; i < tokenIds->size(); i++) {
      token_ids[i] = token_ids_jlong[i];
    }

    std::string decoded = tokenizer_->Decode(token_ids);

    return facebook::jni::make_jstring(decoded);
  }

  jint getVocabSize() {
    return tokenizer_->GetVocabSize();
  }

  facebook::jni::alias_ref<jstring> idToToken(jint id) {
    return facebook::jni::make_jstring(tokenizer_->IdToToken(id));
  }

  jint tokenToId(facebook::jni::alias_ref<jstring> token) {
    return tokenizer_->TokenToId(token->toStdString());
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod(
            "initHybrid", ExecuTorchHuggingFaceTokenizerJni::initHybrid),
        makeNativeMethod(
            "encodeNative", ExecuTorchHuggingFaceTokenizerJni::encode),
        makeNativeMethod(
            "decodeNative", ExecuTorchHuggingFaceTokenizerJni::decode),
        makeNativeMethod(
            "getVocabSizeNative",
            ExecuTorchHuggingFaceTokenizerJni::getVocabSize),
        makeNativeMethod(
            "idToTokenNative", ExecuTorchHuggingFaceTokenizerJni::idToToken),
        makeNativeMethod(
            "tokenToIdNative", ExecuTorchHuggingFaceTokenizerJni::tokenToId),
    });
  }
};
} // namespace executorch::extension

#ifdef EXECUTORCH_BUILD_LLAMA_JNI
extern void register_natives_for_llama();
#else
// No op if we don't build llama
void register_natives_for_llama() {}
#endif
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, [] {
    executorch::extension::ExecuTorchJni::registerNatives();
    register_natives_for_llama();
    executorch::extension::ExecuTorchHuggingFaceTokenizerJni::registerNatives();
  });
}
