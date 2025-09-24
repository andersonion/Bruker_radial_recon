#!/usr/bin/env python3
"""
CuPy installer (v4.2) with papertrail:
- Detect CUDA major (nvidia-smi header or nvcc)
- Purge stale CuPy
- Try CuPy wheels (12x -> 11x)
- If CUDA libs missing (NVRTC/cuFFT/cuBLAS/etc.), install NVIDIA CUDA shim wheels
- **Locate shim lib dirs via importlib.metadata (namespace-safe)** and link *.so into $CONDA_PREFIX/lib
- Add activate hook to export LD_LIBRARY_PATH=$CONDA_PREFIX/lib (and driver paths if needed)
- Verify with a tiny GPU FFT
- Write JSON logs to ./logs/install_cupy-YYYYmmdd-HHMMSS.json and install_cupy-latest.json
- Exit 0 even on CPU-only fallback
"""
import datetime as _dt
import glob, importlib, importlib.metadata as imd, json, os, pathlib, re, shutil, site, subprocess, sys
from typing import List, Dict, Any, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _ts() -> str: return _dt.datetime.now().astimezone().isoformat()
LOG: Dict[str, Any] = {"version": "4.2", "start": _ts(), "steps": []}
def log_step(step: str, **kwargs): LOG["steps"].append({"ts": _ts(), "step": step, **kwargs})
def write_logs(summary: Dict[str, Any]) -> None:
    LOG["end"] = _ts(); LOG["summary"] = summary
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = LOG_DIR / f"install_cupy-{stamp}.json"; latest = LOG_DIR / "install_cupy-latest.json"
    try:
        p.write_text(json.dumps(LOG, indent=2)); latest.write_text(json.dumps(LOG, indent=2))
        summary["log_file"] = str(p)
    except Exception as e:
        summary["log_write_error"] = repr(e)

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has(cmd: str) -> bool: return shutil.which(cmd) is not None

def detect_cuda_major() -> str | None:
    if has("nvidia-smi"):
        r = run(["nvidia-smi"])
        m = re.search(r"CUDA Version:\s*([0-9]+)", r.stdout or "")
        if m: return m.group(1)
        log_step("detect.cuda", method="nvidia-smi", note="no CUDA Version in header")
    if has("nvcc"):
        r = run(["nvcc", "--version"])
        m = re.search(r"release\s+([0-9]+)\.", r.stdout or "")
        if m: return m.group(1)
        log_step("detect.cuda", method="nvcc", note="no release match")
    return None

def pip_install(*pkgs: str) -> bool:
    log_step("pip.install", pkgs=list(pkgs))
    rc = run([sys.executable, "-m", "pip", "install", "-U", *pkgs])
    log_step("pip.install.result", returncode=rc.returncode, stdout=rc.stdout[-4000:], stderr=rc.stderr[-4000:])
    return rc.returncode == 0

def pip_uninstall(*pkgs: str) -> None:
    log_step("pip.uninstall", pkgs=list(pkgs))
    run([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])

def purge_cupy_files() -> None:
    removed = []
    paths = set(site.getsitepackages() + [site.getusersitepackages()])
    for p in paths:
        for pat in ("cupy*", "cupy_cuda*"):
            for x in glob.glob(os.path.join(p, pat)):
                try:
                    if os.path.isdir(x): shutil.rmtree(x); removed.append(x)
                    elif os.path.isfile(x): os.remove(x); removed.append(x)
                except Exception as e:
                    log_step("purge.warn", path=x, err=repr(e))
    log_step("purge.files", removed=removed)

def verify_cupy() -> Tuple[bool, Dict[str, Any]]:
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            x = cp.arange(1024, dtype=cp.float32)
            _ = cp.fft.fft(x)  # cuFFT
            cp.cuda.Stream.null.synchronize()
        return True, {"version": getattr(cp, "__version__", "unknown"), "gpus": int(n)}
    except Exception as e:
        return False, {"err": repr(e)}

def cuda_shim_packages(major_hint: str | None) -> List[str]:
    suf = "cu11" if (major_hint and major_hint.isdigit() and int(major_hint) <= 11) else "cu12"
    return [
        f"nvidia-cuda-runtime-{suf}",
        f"nvidia-cuda-nvrtc-{suf}",
        f"nvidia-cuda-cupti-{suf}",
        f"nvidia-nvjitlink-{suf}",
        f"nvidia-cublas-{suf}",
        f"nvidia-cusolver-{suf}",
        f"nvidia-cusparse-{suf}",
        f"nvidia-cufft-{suf}",
    ]

# mapping of dist name -> expected package resource path for lib dir
DIST_TO_LIBSUB = {
    "nvidia-cuda-runtime-cu12": "nvidia/cuda_runtime/lib",
    "nvidia-cuda-nvrtc-cu12":   "nvidia/cuda_nvrtc/lib",
    "nvidia-cuda-cupti-cu12":   "nvidia/cuda_cupti/lib",
    "nvidia-nvjitlink-cu12":    "nvidia/nvjitlink/lib",
    "nvidia-cublas-cu12":       "nvidia/cublas/lib",
    "nvidia-cusolver-cu12":     "nvidia/cusolver/lib",
    "nvidia-cusparse-cu12":     "nvidia/cusparse/lib",
    "nvidia-cufft-cu12":        "nvidia/cufft/lib",
    "nvidia-cuda-runtime-cu11": "nvidia/cuda_runtime/lib",
    "nvidia-cuda-nvrtc-cu11":   "nvidia/cuda_nvrtc/lib",
    "nvidia-cuda-cupti-cu11":   "nvidia/cuda_cupti/lib",
    "nvidia-nvjitlink-cu11":    "nvidia/nvjitlink/lib",
    "nvidia-cublas-cu11":       "nvidia/cublas/lib",
    "nvidia-cusolver-cu11":     "nvidia/cusolver/lib",
    "nvidia-cusparse-cu11":     "nvidia/cusparse/lib",
    "nvidia-cufft-cu11":        "nvidia/cufft/lib",
}

def collect_shim_lib_dirs() -> List[str]:
    """Namespace-safe: use importlib.metadata to find installed shim lib dirs."""
    dirs: List[str] = []
    for distname, sub in DIST_TO_LIBSUB.items():
        try:
            dist = imd.distribution(distname)
        except imd.PackageNotFoundError:
            continue
        # Use distribution.locate_file to resolve resource path
        path = dist.locate_file(sub)
        if os.path.isdir(path):
            dirs.append(str(path))
        else:
            # fallback: scan site-packages for nvidia/*/lib
            for root in set(site.getsitepackages() + [site.getusersitepackages()]):
                cand = os.path.join(root, sub)
                if os.path.isdir(cand):
                    dirs.append(cand)
    # Dedup while preserving order
    seen=set(); out=[]
    for d in dirs:
        if d not in seen: seen.add(d); out.append(d)
    log_step("shims.libdirs", found=out)
    return out

def link_shims_into_prefix() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"linked": [], "hook_written": False}
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    libdir = os.path.join(prefix, "lib")
    os.makedirs(libdir, exist_ok=True)

    shim_dirs = collect_shim_lib_dirs()
    for d in shim_dirs:
        for src in glob.glob(os.path.join(d, "*.so*")):
            dst = os.path.join(libdir, os.path.basename(src))
            try:
                if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
            except FileNotFoundError:
                pass
            try:
                os.symlink(src, dst); summary["linked"].append(dst)
            except FileExistsError:
                pass

    # write LD_LIBRARY_PATH hook
    actdir = os.path.join(prefix, "etc", "conda", "activate.d")
    deactdir = os.path.join(prefix, "etc", "conda", "deactivate.d")
    os.makedirs(actdir, exist_ok=True); os.makedirs(deactdir, exist_ok=True)
    with open(os.path.join(actdir, "10-cuda-shims.sh"), "w") as f:
        f.write('export _CUDA_SHIMS_OLD_LDLP="${LD_LIBRARY_PATH:-}"\n')
        f.write('export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')
    with open(os.path.join(deactdir, "10-cuda-shims.sh"), "w") as f:
        f.write('export LD_LIBRARY_PATH="${_CUDA_SHIMS_OLD_LDLP:-}"\n'); f.write('unset _CUDA_SHIMS_OLD_LDLP\n')
    summary["hook_written"] = True
    return summary

def find_driver_lib_dirs() -> List[str]:
    candidates = [
        "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib/nvidia", "/lib/x86_64-linux-gnu",
        "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/wsl/lib"
    ]
    return [d for d in candidates if os.path.isdir(d) and glob.glob(os.path.join(d, "libcuda.so*"))]

def add_driver_dirs_to_hook(dirs: List[str]) -> bool:
    if not dirs: return False
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    actdir = os.path.join(prefix, "etc", "conda", "activate.d")
    deactdir = os.path.join(prefix, "etc", "conda", "deactivate.d")
    os.makedirs(actdir, exist_ok=True); os.makedirs(deactdir, exist_ok=True)
    with open(os.path.join(actdir, "11-cuda-driver.sh"), "w") as f:
        f.write('export _CUDA_DRV_OLD_LDLP="${LD_LIBRARY_PATH:-}"\n')
        f.write(f'export LD_LIBRARY_PATH="{":".join(dirs)}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"\n')
        f.write(f'export _CUDA_DRIVER_PATHS="{":".join(dirs)}"\n')
    with open(os.path.join(deactdir, "11-cuda-driver.sh"), "w") as f:
        f.write('export LD_LIBRARY_PATH="${_CUDA_DRV_OLD_LDLP:-}"\n'); f.write('unset _CUDA_DRV_OLD_LDLP\n'); f.write('unset _CUDA_DRIVER_PATHS\n')
    return True

def main() -> int:
    major = detect_cuda_major()
    log_step("detect.result", cuda_major=major, nvidia_smi=shutil.which("nvidia-smi"), nvcc=shutil.which("nvcc"))

    pip_uninstall("cupy", "cupy-cuda12x", "cupy-cuda11x")
    purge_cupy_files()

    candidates = ["cupy-cuda12x", "cupy-cuda11x"] if (not major or int(major) >= 12) else ["cupy-cuda11x", "cupy-cuda12x"]
    LOG["candidates"] = candidates

    tried: List[str] = []
    for pkg in candidates:
        tried.append(pkg)
        if not pip_install(pkg): continue

        ok, meta = verify_cupy()
        log_step("verify.after_wheel", pkg=pkg, ok=ok, meta=meta)
        if ok:
            summary = {"result": "ok", "used": pkg, "cupy": meta}; print(json.dumps(summary)); write_logs(summary); return 0

        # Install shims + link + re-verify if CUDA libs missing
        err = str(meta.get("err", "")).lower()
        if any(k in err for k in ("nvrtc", "libnvrtc", "cufft", "from 'cupy.cuda'", "cublas", "cusolver", "cusparse", "libcuda.so")):
            shims = cuda_shim_packages(major)
            pip_install(*shims)
            link_info = link_shims_into_prefix()
            log_step("shims.installed", packages=shims, link_info=link_info)

            if "libcuda.so" in err:
                drv_dirs = find_driver_lib_dirs()
                added = add_driver_dirs_to_hook(drv_dirs)
                log_step("driver.paths", found=drv_dirs, added_to_hook=added)

            ok2, meta2 = verify_cupy()
            log_step("verify.after_shims", pkg=pkg, ok=ok2, meta=meta2)
            if ok2:
                summary = {"result": "ok", "used": pkg, "cupy": meta2, "cuda_shims": True}; print(json.dumps(summary)); write_logs(summary); return 0

        pip_uninstall(pkg)

    ok_last, meta_last = verify_cupy()
    summary = {"result": "cpu-only", "tried": tried, "verify": meta_last}; print(json.dumps(summary)); write_logs(summary); return 0

if __name__ == "__main__": sys.exit(main())
