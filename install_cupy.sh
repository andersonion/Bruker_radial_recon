#!/usr/bin/env bash
# install_cupy.sh (robust v3.2)
# - Detect NVIDIA/ROCm
# - Try CuPy wheels (12x/11x)
# - If NVRTC missing, install NVIDIA NVRTC/runtime shims and re-verify
# - Verify with a tiny GPU FFT
set -Eeuo pipefail

log(){ >&2 echo "[CuPy] $*"; }
PYTHON_BIN="${PYTHON_BIN:-python}"

detect_cuda_major() {
  # Print ONLY CUDA major digits; no logging here.
  local cuda_ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    if [[ -z "${cuda_ver//N\/A/}" ]]; then
      local header_line
      header_line="$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $0; exit}' || true)"
      cuda_ver="$(sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\(\.[0-9]*\)\?.*/\1/p' <<<"$header_line" || true)"
    fi
  fi
  if [[ -z "$cuda_ver" ]] && command -v nvcc >/dev/null 2>&1; then
    local nvcc_line
    nvcc_line="$(nvcc --version 2>/dev/null | awk -F',|release' '/release/ {gsub(/ /,""); print $3; exit}' || true)"
    cuda_ver="${nvcc_line%%.*}"
  fi
  [[ -n "$cuda_ver" ]] && printf '%s\n' "${cuda_ver%%.*}" || true
}

verify_cupy() {
  # Prints JSON to stdout; returns 0 on success, 1 on failure.
  "$PYTHON_BIN" - <<'PY'
import sys, json
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    if n > 0:
        x = cp.arange(1024, dtype=cp.float32)
        _ = cp.fft.fft(x)
        cp.cuda.Stream.null.synchronize()
    print(json.dumps({"ok": True, "version": getattr(cp,"__version__","unknown"), "gpus": int(n)}))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"ok": False, "err": repr(e)}))
    sys.exit(1)
PY
}

install_nvrtc_runtime() {
  local major="$1"
  if [[ "$major" == "12" ]]; then
    $PYTHON_BIN -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
  elif [[ "$major" == "11" ]]; then
    $PYTHON_BIN -m pip install -U nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11
  else
    $PYTHON_BIN -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 || true
    $PYTHON_BIN -m pip install -U nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11 || true
  fi
}

try_install_nvidia() {
  local major="$1"
  local candidates=()
  if [[ "$major" =~ ^[0-9]+$ ]]; then
    if (( major >= 12 )); then candidates=(cupy-cuda12x cupy-cuda11x)
    else candidates=(cupy-cuda11x cupy-cuda12x); fi
  else
    candidates=(cupy-cuda12x cupy-cuda11x)
  fi

  log "nvidia-smi: $(command -v nvidia-smi || echo not-found)"
  log "Detected CUDA major: ${major:-unknown}"
  log "Candidate wheels: ${candidates[*]}"

  for pkg in "${candidates[@]}"; do
    log "Installing ${pkg}…"
    if ! $PYTHON_BIN -m pip install -U "$pkg"; then
      log "pip install ${pkg} failed; trying next."
      continue
    fi

    # First verify
    out="$(verify_cupy 2>&1)"; status=$?
    if [[ $status -eq 0 ]]; then
      log "✅ ${pkg} works. ${out}"
      return 0
    fi

    log "Import failed after ${pkg}. Inspecting cause…"
    echo "$out" >&2

    if grep -qiE 'libnvrtc\.so\.(11|12)|nvrtc' <<<"$out"; then
      # Determine which major to use for shims (prefer detected, else parse error text)
      local m="$major"
      [[ -z "$m" ]] && m="$(grep -oE 'libnvrtc\.so\.(11|12)' <<<"$out" | grep -oE '(11|12)' | head -n1 || true)"
      log "NVRTC missing → installing NVIDIA NVRTC/runtime shims for CUDA ${m:-unknown}…"
      install_nvrtc_runtime "${m:-}"

      # Re-verify
      out="$(verify_cupy 2>&1)"; status=$?
      if [[ $status -eq 0 ]]; then
        log "✅ ${pkg} works after NVRTC/runtime shims. ${out}"
        return 0
      else
        log "Still failing after NVRTC/runtime; uninstalling ${pkg}."
        $PYTHON_BIN -m pip uninstall -y "$pkg" || true
      fi
    else
      log "Failure wasn’t NVRTC-related; uninstalling ${pkg}."
      $PYTHON_BIN -m pip uninstall -y "$pkg" || true
    fi
  done
  return 1
}

try_install_rocm() {
  command -v rocminfo >/dev/null 2>&1 || return 1
  local rocm_ver
  rocm_ver="$(rocminfo 2>/dev/null | awk -F'[ .:]+' "/Runtime Version:/ {print \$3\".\"\$4; exit}")"
  [[ -z "$rocm_ver" ]] && return 1
  local pkg="cupy-rocm-$(echo "$rocm_ver" | tr '.' '-')"
  log "ROCm ${rocm_ver} detected → installing ${pkg}…"
  if $PYTHON_BIN -m pip install -U "$pkg"; then
    out="$(verify_cupy 2>&1)"; status=$?
    if [[ $status -eq 0 ]]; then
      log "✅ ${pkg} works. ${out}"
      return 0
    else
      log "ROCm verify failed. ${out}"
      return 1
    fi
  fi
  return 1
}

log "Detecting GPU runtime…"
cuda_major="$(detect_cuda_major || true)"

if command -v nvidia-smi >/dev/null 2>&1; then
  if try_install_nvidia "${cuda_major:-}"; then exit 0; fi
fi

if try_install_rocm; then exit 0; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA runtime present but wheels failed. Staying CPU-only."
else
  log "No supported GPU runtime found. Staying CPU-only."
fi
exit 0
