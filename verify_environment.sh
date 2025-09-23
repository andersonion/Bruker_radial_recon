#!/usr/bin/env bash
# verify_environment.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${REPO_DIR}/.envs/mri-recon"

# Initialize conda shell (needed for 'conda activate')
if command -v conda >/dev/null 2>&1; then
  eval "$($(command -v conda) shell.bash hook)"
else
  echo "[verify] ERROR: conda not found in PATH."
  exit 1
fi

echo "[verify] Activating env at ${ENV_PREFIX}"
conda activate "${ENV_PREFIX}"

python - <<'PY'
import sys, json
out = {}
try:
    import numpy as np
    out["numpy"] = np.__version__
    # small compute
    a = np.arange(9, dtype=np.float32).reshape(3,3)
    b = a.T @ a
    out["numpy_ok"] = bool(b.shape==(3,3) and float(b.sum()) > 0)
except Exception as e:
    print("[verify] NumPy FAILED:", e)
    sys.exit(1)

try:
    import nibabel as nib
    import numpy as np
    img = nib.Nifti1Image(np.zeros((4,4,4), dtype=np.float32), affine=np.eye(4))
    _ = img.header  # touch header
    out["nibabel"] = nib.__version__
    out["nibabel_ok"] = True
except Exception as e:
    print("[verify] NiBabel FAILED:", e)
    sys.exit(1)

try:
    import sigpy as sp
    import sigpy.mri as mr
    out["sigpy"] = sp.__version__
    out["sigpy_ok"] = True
except Exception as e:
    print("[verify] SigPy FAILED:", e)
    sys.exit(1)

try:
    import brkraw
    out["brkraw"] = getattr(brkraw, "__version__", "unknown")
    out["brkraw_ok"] = True
except Exception as e:
    print("[verify] Brkraw FAILED:", e)
    sys.exit(1)

# Optional CuPy test
cupy_status = {"installed": False, "gpu_count": 0, "ok": False}
try:
    import cupy as cp
    cupy_status["installed"] = True
    try:
        n = cp.cuda.runtime.getDeviceCount()
        cupy_status["gpu_count"] = int(n)
        # tiny op to confirm kernels run (if a device exists)
        x = cp.arange(1024, dtype=cp.float32)
        y = cp.fft.fft(x)  # exercises cuFFT
        cp.cuda.Stream.null.synchronize()
        cupy_status["ok"] = True
    except Exception as e:
        cupy_status["ok"] = False
except Exception:
    pass

print(json.dumps({"packages": out, "cupy": cupy_status}, indent=2))
PY

echo "[verify] Environment looks good."
