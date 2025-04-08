/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import org.pytorch.executorch.annotations.Experimental;
import com.facebook.soloader.nativeloader.SystemDelegate;

@Experimental
public class HuggingFaceTokenizer {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(String jsonPath);

  public HuggingFaceTokenizer(String jsonPath) {
        mHybridData = initHybrid(jsonPath);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  public int[] encode(String text) {
    return encodeNative(text);
  };

  public String decode(int[] tokenIds) {
    return decode(tokenIds, false);
  };

  public String decode(int[] tokenIds, boolean skipSpecialTokens) {
    return decodeNative(tokenIds, skipSpecialTokens);
  };

  public int getVocabSize() {
    return getVocabSizeNative();
  };

  public String idToToken(int id){
    return idToTokenNative(id);
  };

  public int tokenToId(String token){
    return tokenToIdNative(token);
  };

  @DoNotStrip
  private native int[] encodeNative(String text);

  @DoNotStrip
  private native String decodeNative(int[] tokenIds, boolean skipSpecialTokens);
  
  @DoNotStrip 
  private native int getVocabSizeNative();

  @DoNotStrip
  private native String idToTokenNative(int id);

  @DoNotStrip
  private native int tokenToIdNative(String token);
}