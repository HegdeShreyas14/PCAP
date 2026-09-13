"""Capture a reproducible-relevant environment fingerprint as JSON."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

# Some managed Windows sessions cannot write to the user Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))


def _version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/environment.json")
    args = ap.parse_args()
    packages = ["numpy", "scipy", "networkx", "numba", "pandas",
                "matplotlib", "gglasso", "torch"]
    data = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.node(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "packages": {name: _version(name) for name in packages},
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    try:
        import torch
        data["cuda_available"] = torch.cuda.is_available()
        data["cuda_runtime"] = torch.version.cuda
        data["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        data.update(cuda_available=False, cuda_runtime=None, gpu=None)
    try:
        data["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:
        data["nvidia_smi"] = None
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
