"""
provenance.py — record exactly which code and settings produced a dataset.

Every harness writes `<out>_env.json` next to its CSV via write_env(). Without
it we could not tell which code version produced percolation.csv (bug 8), and
that file ended up irreproducible.

Also provides s_fingerprint(): two runs that are meant to be paired (e.g. 1t
vs 8t) must see the same S, and comparing these columns proves they did.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cmd(*argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def system_snapshot():
    """
    Hardware, OS, BLAS and power state, best effort (Linux). Recorded with
    every run because two of our bugs were environmental: BLAS behaviour
    (bug 8) and an unrecorded power profile plus competing load (bug 9).
    """
    cpuinfo = _read("/proc/cpuinfo") or ""
    field = lambda k: next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines()
                            if l.startswith(k)), None)
    meminfo = _read("/proc/meminfo") or ""
    mem_kb = next((int(l.split()[1]) for l in meminfo.splitlines()
                   if l.startswith("MemTotal")), None)
    osrel = dict(l.split("=", 1) for l in (_read("/etc/os-release") or "").splitlines() if "=" in l)
    cf = "/sys/devices/system/cpu/cpu0/cpufreq/"
    snap = {
        "machine_vendor": _read("/sys/class/dmi/id/sys_vendor"),
        "machine_product": _read("/sys/class/dmi/id/product_name"),
        "bios_version": _read("/sys/class/dmi/id/bios_version"),
        "cpu_model": field("model name"),
        "cpu_physical_cores": int(field("cpu cores")) if field("cpu cores") else None,
        "cpu_logical_cores": os.cpu_count(),
        "smt_active": _read("/sys/devices/system/cpu/smt/active"),
        "ram_gib": round(mem_kb / 2**20, 1) if mem_kb else None,
        "os": osrel.get("PRETTY_NAME", "").strip('"') or None,
        "kernel": platform.release(),
        "libc": " ".join(platform.libc_ver()),
        "cpufreq_driver": _read(cf + "scaling_driver"),
        "cpufreq_governor": _read(cf + "scaling_governor"),
        "cpufreq_epp": _read(cf + "energy_performance_preference"),
        "power_profile": _cmd("powerprofilesctl", "get"),
        "ac_online": _cmd("sh", "-c", "cat /sys/class/power_supply/*/online | tr '\\n' ' '"),
        "loadavg_at_start": _read("/proc/loadavg"),
        "docker_running": _cmd("docker", "ps", "--format", "{{.Names}}"),
    }
    try:
        import numpy as np
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        snap["blas_library"] = f"{blas.get('name')} {blas.get('version')}"
    except Exception:
        snap["blas_library"] = None
    smi = _cmd("nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
               "--format=csv,noheader")
    snap["nvidia_smi"] = smi
    return snap


def run_metadata(args=None):
    """Code version, environment and arguments for one harness run."""
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "src")
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": vars(args) if args is not None else None,
        "git_commit": _git("rev-parse", "HEAD"),
        # A dirty src/ means the commit hash does not describe the code that
        # ran. Datasets for the paper must have src_dirty == false.
        "src_dirty": bool(dirty) if dirty is not None else None,
        "src_md5": {f: hashlib.md5(open(os.path.join(SRC, f), "rb").read()).hexdigest()
                    for f in sorted(os.listdir(SRC)) if f.endswith(".py")},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cores": os.cpu_count(),
        "blas_threads": {v: os.environ.get(v) for v in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "networkx": _version("networkx"),
        "gglasso": _version("gglasso"),
        "system": system_snapshot(),
    }
    return meta


def env_path_for(out):
    return (out[:-4] if out.endswith(".csv") else out) + "_env.json"


def write_env(out, extra=None, args=None):
    """Write <out>_env.json and return its path."""
    meta = run_metadata(args)
    if extra:
        meta.update(extra)
    path = env_path_for(out)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if meta["src_dirty"]:
        print(f"!! WARNING: src/ has uncommitted changes; {path} records "
              f"git_commit={meta['git_commit']} but that is not the code that ran.")
    return path


def s_fingerprint(S):
    """
    Thread-robust fingerprint of a sample covariance. Not a bitwise hash: S is
    identical across BLAS thread counts only to ~1e-12, so compare these with a
    relative tolerance (1e-9 is ample) rather than for equality.
    """
    import numpy as np
    return {"S_trace": float(np.trace(S)),
            "S_fro": float(np.linalg.norm(S)),
            "S_sum": float(S.sum())}
