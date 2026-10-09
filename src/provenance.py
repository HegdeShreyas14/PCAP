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
