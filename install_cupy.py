#!/usr/bin/env python3
import json, os, re, sys, glob, shutil, subprocess, importlib, pathlib

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has(cmd):
    return shutil.which(cmd) is not None

def detect_cuda_major():
    # Parse header from nvidia-smi (works on older drivers); fallback to nvcc
    if has("nvidia-smi"):
        r = run(["nvidia-smi"])
        m = re.search(r"CUDA Version:\s*([0-9]+)", r.stdout)
        if m:
            return m.group(1)
    if has("nvcc"):
        r = run(["nvcc", "--version"])
        m = re.search(r"release\s+([0-9]+)\.", r.stdout)
        if m:
            return m.group(1)
    return None

def pip_install(*pkgs):
    return run([sys.executable, "-m", "pip", "install", "-U", *pkgs]).returncode == 0

def pip_uninstall(*pkgs):
    run([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])

def verify_cupy():
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            x = cp.arange(1024, dtype=cp.float32)
            _ = cp.fft.fft(x)
            cp.cuda.Stream.null.synchronize()
        return True, {"version": getattr(cp, "__version__", "unknown"), "gpus": int(n)}
    except Exception as e:
        return False, {"err": repr(e)}

def install_nvrtc_runtime(major):
    # Install NVIDIA runtime/NVRTC shim wheels from PyPI
    if major == "12":
        pkgs = ["nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"]
    elif major == "11":
        pkgs = ["nvidia-cuda-runtime-cu11", "nvidia-cuda-nvrtc-cu11"]
    else:
        pkgs = ["nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12",
                "nvidia-cuda-runtime-cu11", "nvidia-cuda-nvrtc-cu11"]
    pip_install(*pkgs)
    link_shims_into_prefix()

def link_shims_into_prefix():
    # Symlink shim .so files into $CONDA_PREFIX/lib and add an activate hook
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    libdir = os.path.join(prefix, "lib")
    os.makedirs(libdir, exist_ok=True)

    shim_dirs = []
    for name in ("nvidia.cuda_nvrtc", "nvidia.cuda_runtime"):
        try:
            m = importlib.import_module(name)
            d = os.path.join(os.path.dirname(m.__file__), "lib")
            if os.path.isdir(d):
                shim_dirs.append(d)
        except Exception:
            pass

    for d in shim_dirs:
        for src in glob.glob(os.path.join(d, "*.so*")):
            dst = os.path.join(libdir, os.path.basename(src))
            try:
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
            except FileNotFoundError:
                pass
            os.symlink(src, dst)

    actdir = os.path.join(prefix, "etc", "conda", "activate.d")
    deactdir = os.path.join(prefix, "etc", "conda", "deactivate.d")
    os.makedirs(actdir, exist_ok=True)
    os.makedirs(deactdir, exist_ok=True)
    with open(os.path.join(actdir, "10-cuda-shims.sh"), "w") as f:
        f.write('export _CUDA_SHIMS_OLD_LDLP="${LD_LIBRARY_PATH:-}"\n')
        f.write('export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')
    with open(os.path.join(deactdir, "10-cuda-shims.sh"), "w") as f:
        f.write('export LD_LIBRARY_PATH="${_CUDA_SHIMS_OLD_LDLP:-}"\n')
        f.write('unset _CUDA_SHIMS_OLD_LDLP\n')

def main():
    info = {}
    major = detect_cuda_major()
    info["detected_cuda_major"] = major

    if major and major.isdigit() and int(major) >= 12:
        candidates = ["cupy-cuda12x", "cupy-cuda11x"]
    else:
        candidates = ["cupy-cuda11x", "cupy-cuda12x"]

    tried = []
    for pkg in candidates:
        tried.append(pkg)
        pip_install(pkg)
        ok, meta = verify_cupy()
        if ok:
            info.update({"result": "ok", "cupy": meta, "used": pkg})
            print(json.dumps(info))
            return 0

        err = str(meta.get("err", "")).lower()
        if "nvrtc" in err or "libnvrtc.so" in err:
            install_nvrtc_runtime(major or "")
            ok2, meta2 = verify_cupy()
            if ok2:
                info.update({"result": "ok", "cupy": meta2, "used": pkg, "nvrtc_fix": True})
                print(json.dumps(info))
                return 0

        # If we got here, this wheel didn't work; remove and try next
        pip_uninstall(pkg)

    info.update({"result": "cpu-only", "tried": tried, "verify": verify_cupy()[1]})
    print(json.dumps(info))
    return 0

if __name__ == "__main__":
    sys.exit(main())
