#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds mlx.metallib for a given Apple SDK directly from the MLX kernel
# sources.
#
# The MLX submodule's CMake builds the metallib via add_custom_target, which the
# CMake Xcode generator (used for ExecuTorch's iOS/macOS framework builds) does
# not schedule. This script reproduces that build so the metallib is produced
# for any generator. It parses the build_kernel(...) calls from MLX's
# kernels/CMakeLists.txt so the AOT kernel set stays in sync across submodule
# bumps (only the MLX_METAL_JIT=ON set — the kernels compiled unconditionally).
#
# Usage:
#   build_metallib.sh --sdk <macosx|iphoneos|iphonesimulator> \
#                     --version-min <flag=value> \
#                     --output <path/to/mlx.metallib>
#
# Example (iOS device):
#   build_metallib.sh --sdk iphoneos \
#                     --version-min -mios-version-min=17.0 \
#                     --output cmake-out/mlx.metallib

set -euo pipefail

SDK="macosx"
VERSION_MIN=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk) SDK="$2"; shift 2 ;;
    --version-min) VERSION_MIN="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "error: --output is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_ROOT="${SCRIPT_DIR}/../third-party/mlx"
KERNELS_DIR="${MLX_ROOT}/mlx/backend/metal/kernels"
KERNELS_CMAKE="${KERNELS_DIR}/CMakeLists.txt"

if [[ ! -f "$KERNELS_CMAKE" ]]; then
  echo "error: MLX kernels CMakeLists not found at ${KERNELS_CMAKE}" >&2
  echo "       Run: git submodule update --init backends/mlx/third-party/mlx" >&2
  exit 1
fi

# Extract the always-on AOT kernel list: top-level build_kernel(NAME ...) calls
# that appear before the first if(...) block. This deliberately excludes
# version-gated kernels (e.g. fence, behind if(MLX_METAL_VERSION ...)) and the
# non-JIT-only kernels (behind if(NOT MLX_METAL_JIT)), matching what an
# MLX_METAL_JIT=ON build compiles unconditionally.
KERNELS=$(awk '
  /^[[:space:]]*build_kernel\(/ {
    seen = 1
    line = $0
    sub(/^[[:space:]]*build_kernel\([[:space:]]*/, "", line)
    sub(/[[:space:]].*/, "", line)
    sub(/\).*/, "", line)
    print line
    next
  }
  # Once kernels have started, the first if(...) block ends the always-on set.
  seen && /^[[:space:]]*if\(/ { exit }
' "$KERNELS_CMAKE")

if [[ -z "$KERNELS" ]]; then
  echo "error: no build_kernel() entries parsed from ${KERNELS_CMAKE}" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

METAL_FLAGS=(-x metal -Wall -Wextra -fno-fast-math
             -Wno-c++17-extensions -Wno-c++20-extensions)
if [[ -n "$VERSION_MIN" ]]; then
  METAL_FLAGS+=("$VERSION_MIN")
fi

AIR_FILES=()
echo "Building mlx.metallib for SDK '${SDK}' from kernels:"
while IFS= read -r kernel; do
  [[ -z "$kernel" ]] && continue
  src="${KERNELS_DIR}/${kernel}.metal"
  if [[ ! -f "$src" ]]; then
    echo "  - skip ${kernel} (no ${kernel}.metal)"
    continue
  fi
  air="${WORK_DIR}/${kernel}.air"
  echo "  - ${kernel}"
  xcrun -sdk "$SDK" metal "${METAL_FLAGS[@]}" -c "$src" \
    -I "$MLX_ROOT" -o "$air"
  AIR_FILES+=("$air")
done <<< "$KERNELS"

mkdir -p "$(dirname "$OUTPUT")"
xcrun -sdk "$SDK" metallib "${AIR_FILES[@]}" -o "$OUTPUT"
echo "Wrote ${OUTPUT}"
