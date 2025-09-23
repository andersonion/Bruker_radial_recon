#!/usr/bin/env bash
# install_cupy.sh (robust v3)
# - Detects NVIDIA/ROCm
# - Installs matching CuPy wheel
# - If NVRTC missing, installs NVIDIA runtime/NVRTC shim pkgs
# - Verifies with a tiny GPU FFT
set -Eeuo pipefail

log(){ >&2 echo "[CuPy] $*"; }
PYTHON_BIN="${PYTHON_BIN:-python}"

detect_cuda_major() {
  # Return ONLY the CUDA major version (digits) to stdout; no logging here.
  local cuda_ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Newer drivers support --query; older may not.
    cuda_ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    if [[ -z "${cuda_ver//N\/A/}" ]]; then
      # Fallback: parse header line
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
  # Emit only digits (or nothing)
  [[ -n "$cuda_ver" ]] && printf '%s\n' "${cuda_ver%%.*}" || true
}

verify_cupy_py='
import sys, json
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    # Light GPU op if at least one device exists
    if n > 0:
        x = cp.arange(1024, dtype=cp.float32)
        _ = cp.fft.fft(x)
        cp.cuda.Stream.null.synchronize()
    print(json.dumps({"ok": True, "version": getattr(cp, "__version__", "unknown"), "gpus": int(n)}))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"ok": False, "err": repr(e)}))
    sys.exit(1)
'

install_nvrtc_runtime() {
  local major="$1"
  if [[ "$major" == "12" ]]; then
    # These are NVIDIA’s runtime shims on PyPI
    $PYTHON_BIN -m pip install -U "nvidia-cuda-runtime-cu12" "nvidia-cuda-nvrtc-cu12"
  elif [[ "$major" == "11" ]]; then
    $PYTHON_BIN -m pip install -U "nvidia-cuda-runtime-cu11" "nvidia-cuda-nvrtc-cu11"
  else
    # Try 12 first, then 11
    $PYTHON_BIN -m pip install -U "nvidia-cuda-runtime-cu12" "nvidia-cuda-nvrtc-cu12" || true
    $PYTHON_BIN -m pip install -U "nvidia-cuda-runtime-cu11" "nvidia-cuda-nvrtc-cu11" || true
  fi
}

try_install_nvidia() {
  local major="$1"
  local candidates=()
  if [[ "$major" =~ ^[0-9]+$ ]]; then
    (( major >= 12 )) && candidates=(cupy-cuda12x cupy-cuda11x) || candidates=(cupy-cuda11x cupy-cuda12x)
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

    local out
    if out="$($PYTHON_BIN - <<PY 2>&1)"; then
${verify_cupy_py}
PY
    then
      log "✅ $(echo "$out" | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p') works."
      echo "$out"
      return 0
    else
      log "Import failed after ${pkg}. Inspecting cause…"
      echo "$out" >&2
      if grep -qE 'libnvrtc\.so\.(11|12)|nvrtc' <<<"$out"; then
        local m="$major"
        [[ -z "$m" ]] && m="$(grep -oE 'libnvrtc\.so\.(11|12)' <<<"$out" | grep -oE '(11|12)' | head -n1 || true)"
        log "NVRTC missing; installing NVIDIA NVRTC/runtime shims for CUDA ${m:-unknown}…"
        install_nvrtc_runtime "${m:-}"
        # Re-verify
        if out="$($PYTHON_BIN - <<PY 2>&1)"; then
${verify_cupy_py}
PY
        then
          log "✅ ${pkg} works after installing NVRTC/runtime shims."
          echo "$out"
          return 0
        else
          log "Still failing after NVRTC/runtime shims; uninstalling ${pkg}."
          $PYTHON_BIN -m pip uninstall -y "$pkg" || true
        fi
      else
        log "Failure was not NVRTC-related; uninstalling ${pkg}."
        $PYTHON_BIN -m pip uninstall -y "$pkg" || true
      fi
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
    $PYTHON_BIN - <<PY || { log "ROCm verify failed."; return 1; }
${verify_cupy_py}
PY
    log "✅ ${pkg} works."
    return 0
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
