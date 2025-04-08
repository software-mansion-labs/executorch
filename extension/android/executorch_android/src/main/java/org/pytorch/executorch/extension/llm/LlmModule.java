/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import org.pytorch.executorch.annotations.Experimental;

/**
 * LlmModule is a wrapper around the Executorch LLM. It provides a simple interface to
 * generate text from the model.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class LlmModule {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;
  private static final int DEFAULT_SEQ_LEN = 2048;
  private static final boolean DEFAULT_ECHO = false;

  @DoNotStrip
  private static native HybridData initHybrid(String modulePath, String tokenizerPath, float temperature, String dataPath);

  /** Constructs a LLM Module for a model with given model path, tokenizer, temperature. */
  public LlmModule(String modulePath, String tokenizerPath, float temperature) {
    mHybridData = initHybrid(modulePath, tokenizerPath, temperature, null);
  }

  /**
   * Constructs a LLM Module for a model with given model path, tokenizer, temperature and data
   * path.
   */
  public LlmModule(String modulePath, String tokenizerPath, float temperature, String dataPath) {
    mHybridData = initHybrid(modulePath, tokenizerPath, temperature, dataPath);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results.
   */
  public int generate(String prompt, LlmCallback llmCallback) {
    return generate(prompt, DEFAULT_SEQ_LEN, llmCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   */
  public int generate(String prompt, int seqLen, LlmCallback llmCallback) {
    return generate(null, 0, 0, 0, prompt, seqLen, llmCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, LlmCallback llmCallback, boolean echo) {
    return generate(null, 0, 0, 0, prompt, DEFAULT_SEQ_LEN, llmCallback, echo);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, int seqLen, LlmCallback llmCallback, boolean echo) {
    return generate(null, 0, 0, 0, prompt, seqLen, llmCallback, echo);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  @DoNotStrip
  public native int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo);

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
