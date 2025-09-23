#!/usr/bin/env bash
# setup_env.sh
# Create/update a project-local conda env in ./.envs/mri-recon and run the CuPy installer.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${REPO_DIR}/.envs/mri-recon"
ENV_FILE="${REPO_DIR}/environment.yml"

echo "[setup] Repo: ${REPO_DIR}"
echo "[setup] Env prefix: ${ENV_PREFIX}"

# Choose conda/mamba
if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  echo "[setup] ERROR: conda/mamba not found in PATH. Install Miniconda/Mambaforge, then re-run."
  exit 1
fi
echo "[setup] Using solver: ${CONDA_CMD}"

mkdir -p "${REPO_DIR}/.envs"

# Provide a default environment.yml if missing
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[setup] No environment.yml found; writing a safe default."
  cat > "${ENV_FILE}" <<'YAML'
name: mri-recon
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - nibabel
  - scipy
  - numba
  - pywavelets
  - tqdm
  - pip
  - pip:
      - sigpy
      - brkraw
YAML
fi

# Initialize conda shell
if command -v conda >/dev/null 2>&1; then
  eval "$($(command -v conda) shell.bash hook)"
fi

# Create or update the env by prefix (inside repo)
if [[ -d "${ENV_PREFIX}" ]]; then
  echo "[setup] Updating existing env at ${ENV_PREFIX}"
  ${CONDA_CMD} env update --prefix "${ENV_PREFIX}" -f "${ENV_FILE}" --prune
else
  echo "[setup] Creating env at ${ENV_PREFIX}"
  ${CONDA_CMD} env create  --prefix "${ENV_PREFIX}" -f "${ENV_FILE}"
fi

# Activate and run CuPy installer
echo "[setup] Activating env..."
conda activate "${ENV_PREFIX}"

# Write the CuPy installer next to this script (idempotent)
CUPY_INSTALL="${REPO_DIR}/install_cupy.sh"
if [[ ! -f "${CUPY_INSTALL}" ]]; then
  cat > "${CUPY_INSTALL}" <<'BASH'
#!/usr/bin/env bash
# install_cupy.sh
# Detects GPU runtime and installs a matching CuPy wheel if possible.

set -euo pipefail

echo "[CuPy] Detecting GPU runtime..."
# macOS note: no NVIDIA/ROCm; this will fall through and keep CPU-only.
UNAME_S="$(uname -s || true)"

if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_LINE="$(nvidia-smi | awk '/CUDA Version/ {print $0; exit}' || true)"
  CUDA_MAJOR="$(echo "${CUDA_LINE}" | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p')"
  if [[ -n "${CUDA_MAJOR:-}" ]]; then
    if [[ "${CUDA_MAJOR}" -ge 12 ]]; then
      echo "[CuPy] NVIDIA + CUDA ${CUDA_MAJOR} detected → installing cupy-cuda12x"
      python -m pip install --upgrade "cupy-cuda12x"
    else
      echo "[CuPy] NVIDIA + CUDA ${CUDA_MAJOR} detected → installing cupy-cuda11x"
      python -m pip install --upgrade "cupy-cuda11x"
    fi
    python - <<'PY'
try:
    import cupy as cp
    print("[CuPy] Installed:", cp.__version__, "| GPUs:", cp.cuda.runtime.getDeviceCount())
except Exception as e:
    print("[CuPy] Import failed after install:", e)
    raise
PY
    exit 0
  fi
fi

# ROCm (Linux + AMD)
if command -v rocminfo >/dev/null 2>&1; then
  ROCM_VER="$(rocminfo 2>/dev/null | awk -F'[ .:]+' '/Runtime Version:/ {print $3"."$4; exit}')"
  if [[ -n "${ROCM_VER:-}" ]]; then
    PKG="cupy-rocm-$(echo "${ROCM_VER}" | tr '.' '-')"
    echo "[CuPy] AMD ROCm ${ROCM_VER} detected → installing ${PKG} (if available)"
    if python -m pip install --upgrade "${PKG}"; then
      python - <<'PY'
try:
    import cupy as cp
    print("[CuPy] Installed:", cp.__version__, "| GPUs:", cp.cuda.runtime.getDeviceCount())
except Exception as e:
    print("[CuPy] Import failed after install:", e)
    raise
PY
      exit 0
    else
      echo "[CuPy] ${PKG} not available for your setup. Staying CPU-only."
      exit 0
    fi
  fi
fi

echo "[CuPy] No supported GPU runtime found (or unsupported OS). Staying CPU-only."
BASH
  chmod +x "${CUPY_INSTALL}"
fi

echo "[setup] Running ${CUPY_INSTALL} ..."
bash "${CUPY_INSTALL}" || {
  echo "[setup] CuPy step failed or skipped; continuing with CPU-only."
}

echo "[setup] Done. To use the env:"
echo "    conda activate ${ENV_PREFIX}"
