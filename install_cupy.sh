#!/usr/bin/env bash
# install_cupy.sh (robust v3.3)
# - Detect NVIDIA/ROCm (parse nvidia-smi header; no --query)
# - Install CuPy (12x/11x)
# - If NVRTC missing, install NVIDIA NVRTC/runtime shims,
#   link their libs into $CONDA_PREFIX/lib, and add an activate hook
# - Verify with a tiny GPU FFT
set -Eeuo pipefail

log(){ >&2 echo "[CuPy] $*"; }
PYTHON_BIN="${PYTHON_BIN:-python}"

detect_cuda_major() {
  # Print ONLY CUDA major digits; no logging here.
  local header_line cuda_ver
  if command -v nvidia-smi >/dev/null 2>&1; then
    header_line="$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $0; exit}' || true)"
    cuda_ver="$(sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\(\.[0-9]*\)\?.*/\1/p' <<<"$header_line" || true)"
  fi
  if [[ -z "$cuda_ver" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_ver="$(nvcc --version 2>/dev/null | awk -F',|release' '/release/ {gsub(/ /,""); print $3; exit}')"
    cuda_ver="${cuda_ver%%.*}"
  fi
  [[ "$cuda_ver" =~ ^[0-9]+$ ]] && printf '%s\n' "$cuda_ver" || true
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

link_nvidia_shims_into_prefix() {
  # Symlink NVRTC/runtime .so's from the PyPI shim packages into $CONDA_PREFIX/lib,
  # and add a simple activate hook to append that path to LD_LIBRARY_PATH.
  local prefix="${CONDA_PREFIX:-}"
  if [[ -z "$prefix" ]]; then
    prefix="$("$PYTHON_BIN" - <<'PY'
import os
print(os.environ.get("CONDA_PREFIX",""))
PY
)"
  fi
  if [[ -z "$prefix" ]]; then
    log "CONDA_PREFIX not set; cannot link shims."
    return 1
  fi

  local libdirs
  libdirs="$("$PYTHON_BIN" - <<'PY'
import os, importlib
dirs=[]
for name in ("nvidia.cuda_nvrtc","nvidia.cuda_runtime"):
    try:
        m=importlib.import_module(name)
        d=os.path.join(os.path.dirname(m.__file__), "lib")
        if os.path.isdir(d): dirs.append(d)
    except Exception:
        pass
print(":".join(dirs))
PY
)"
  if [[ -z "$libdirs" ]]; then
    log "No NVRTC/runtime shim lib dirs found in site-packages."
    return 1
  fi

  mkdir -p "$prefix/lib"
  IFS=':' read -r -a arr <<< "$libdirs"
  for d in "${arr[@]}"; do
    for so in "$d"/*.so*; do
      [[ -e "$so" ]] || continue
      ln -sf "$so" "$prefix/lib/$(basename "$so")"
    done
  done

  # Minimal activate/deactivate hooks: prepend $CONDA_PREFIX/lib to LD_LIBRARY_PATH
  mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
  cat > "$prefix/etc/conda/activate.d/10-cuda-shims.sh" <<'ACT'
# added by install_cupy.sh
export _CUDA_SHIMS_OLD_LDLP="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACT
  cat > "$prefix/etc/conda/deactivate.d/10-cuda-shims.sh" <<'DEACT'
# added by install_cupy.sh
export LD_LIBRARY_PATH="${_CUDA_SHIMS_OLD_LDLP:-}"
unset _CUDA_SHIMS_OLD_LDLP
DEACT
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
  link_nvidia_shims_into_prefix || true
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

    out="$(verify_cupy 2>&1)"; status=$?
    if [[ $status -eq 0 ]]; then
      log "✅ ${pkg} works. ${out}"
      return 0
    fi

    log "Import failed after ${pkg}. Inspecting cause…"
    echo "$out" >&2

    if grep -qiE 'libnvrtc\.so\.(11|12)|nvrtc' <<<"$out"; then
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
    } else
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
