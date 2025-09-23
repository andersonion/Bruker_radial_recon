#!/usr/bin/env bash
set -euo pipefail

echo "[CuPy installer] Detecting GPU runtime..."
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_LINE=$(nvidia-smi | awk '/CUDA Version/ {print $0; exit}')
  CUDA_MAJOR=$(echo "$CUDA_LINE" | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p')
  if [ -n "${CUDA_MAJOR:-}" ]; then
    if [ "$CUDA_MAJOR" -ge 12 ]; then
      echo "[CuPy installer] NVIDIA + CUDA $CUDA_MAJOR detected → installing cupy-cuda12x"
      python -m pip install --upgrade "cupy-cuda12x"
    else
      echo "[CuPy installer] NVIDIA + CUDA $CUDA_MAJOR detected → installing cupy-cuda11x"
      python -m pip install --upgrade "cupy-cuda11x"
    fi
    exit 0
  fi
fi

if command -v rocminfo >/dev/null 2>&1; then
  ROCM_VER=$(rocminfo 2>/dev/null | awk -F'[ .:]+' '/Runtime Version:/ {print $3"."$4; exit}')
  if [ -n "${ROCM_VER:-}" ]; then
    PKG="cupy-rocm-$(echo "$ROCM_VER" | tr '.' '-')"
    echo "[CuPy installer] AMD ROCm $ROCM_VER detected → installing ${PKG} (if available)"
    python -m pip install --upgrade "$PKG" || {
      echo "[CuPy installer] ${PKG} not available for your setup. Check CuPy ROCm wheels for your version."
    }
    exit 0
  fi
fi

echo "[CuPy installer] No supported GPU runtime found. Staying CPU-only (NumPy)."
