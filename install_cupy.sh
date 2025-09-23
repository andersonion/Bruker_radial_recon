#!/usr/bin/env bash
# Robust CuPy installer: NVIDIA -> try 12x then 11x (based on detected CUDA),
# verify import + device access; uninstall and fallback if needed. ROCm supported.
set -Eeuo pipefail

log(){ echo "[CuPy] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

detect_cuda_major() {
  local cuda_ver=""
  # Preferred (structured) query, if supported by driver:
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Show where it came from (helps when PATH is weird)
    log "nvidia-smi at: $(command -v nvidia-smi)"
    # Try --query (newer drivers)
    cuda_ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    # Fallback: parse header
    if [[ -z "${cuda_ver//N\/A/}" ]]; then
      local header_line
      header_line="$(nvidia-smi | awk '/CUDA Version/ {print $0; exit}' || true)"
      cuda_ver="$(sed -n 's/.*CUDA Version: \([0-9][0-9]*\(\.[0-9]*\)\?\).*/\1/p' <<<"$header_line" || true)"
    fi
  fi
  # Fallback: nvcc (toolkit) if present
  if [[ -z "${cuda_ver}" ]] && command -v nvcc >/dev/null 2>&1; then
    local nvcc_line
    nvcc_line="$(nvcc --version | awk -F',|release' '/release/ {gsub(/ /,""); print $3; exit}' || true)"
    cuda_ver="$nvcc_line"
  fi
  # Return just the major
  if [[ -n "${cuda_ver}" ]]; then
    echo "${cuda_ver%%.*}"
    return 0
  fi
  echo ""
  return 1
}

verify_cupy_python='
import sys
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    # Light compute to ensure kernels run (if device exists)
    if n > 0:
        x = cp.arange(1024, dtype=cp.float32)
        _ = cp.fft.fft(x)
        cp.cuda.Stream.null.synchronize()
    print(f"[CuPy] OK: {cp.__version__} | GPUs: {n}")
    sys.exit(0)
except Exception as e:
    print("[CuPy] import/use failed:", repr(e))
    sys.exit(1)
'

try_install_nvidia() {
  local major="$1"
  local candidates=()
  if [[ -n "$major" ]]; then
    if (( major >= 12 )); then
      candidates=(cupy-cuda12x cupy-cuda11x)
    elif (( major >= 11 )); then
      candidates=(cupy-cuda11x cupy-cuda12x)
    else
      candidates=(cupy-cuda12x cupy-cuda11x)
    fi
  else
    candidates=(cupy-cuda12x cupy-cuda11x)
  fi

  log "NVIDIA runtime detected (CUDA major='${major:-unknown}'). Candidates: ${candidates[*]}"
  for pkg in "${candidates[@]}"; do
    log "Attempting: ${pkg}"
    if "${PYTHON_BIN}" -m pip install -U "${pkg}"; then
      if "${PYTHON_BIN}" - <<PY
${verify_cupy_python}
PY
      then
        log "✅ ${pkg} works."
        return 0
      else
        log "❌ ${pkg} installed but failed to initialize; uninstalling."
        "${PYTHON_BIN}" -m pip uninstall -y "${pkg}" || true
      fi
    else
      log "pip install ${pkg} failed; trying next."
    fi
  done
  return 1
}

try_install_rocm() {
  if ! command -v rocminfo >/dev/null 2>&1; then
    return 1
  fi
  local rocm_ver
  rocm_ver="$(rocminfo 2>/dev/null | awk -F'[ .:]+' "/Runtime Version:/ {print \$3\".\"\$4; exit}")"
  if [[ -z "${rocm_ver}" ]]; then
    return 1
  fi
  local pkg="cupy-rocm-$(echo "${rocm_ver}" | tr '.' '-')"
  log "ROCm runtime detected (${rocm_ver}); attempting ${pkg}"
  if "${PYTHON_BIN}" -m pip install -U "${pkg}"; then
    if "${PYTHON_BIN}" - <<PY
${verify_cupy_python}
PY
    then
      log "✅ ${pkg} works."
      return 0
    else
      log "❌ ${pkg} installed but failed to initialize; uninstalling."
      "${PYTHON_BIN}" -m pip uninstall -y "${pkg}" || true
    fi
  else
    log "pip install ${pkg} failed."
  fi
  return 1
}

log "Detecting GPU runtime..."
NV=1
if command -v nvidia-smi >/dev/null 2>&1; then
  cuda_major="$(detect_cuda_major || true)"
  if try_install_nvidia "${cuda_major:-}"; then
    exit 0
  else
    NV=0
  fi
fi

if try_install_rocm; then
  exit 0
fi

if (( NV == 1 )); then
  log "NVIDIA runtime present but wheels failed. Staying CPU-only."
else
  log "No supported GPU runtime found. Staying CPU-only."
fi
exit 0
