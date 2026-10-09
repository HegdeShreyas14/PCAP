"""
Regression test: the sampled covariance S must not depend on the BLAS thread
count. If it does, the 1t vs 8t sweeps are not a paired comparison (this
happened once: see ProgressLog, bug 8).

BLAS threads can only be set before numpy is imported, so each thread count
runs in its own subprocess.

    python tests/test_sampling_determinism.py      # or: pytest tests/
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = sys.argv[1]
sys.path.insert(0, sys.argv[3])
import numpy as np
from blockgen import make_instance, REGIMES
out = {}
for r in REGIMES:
    for p in (100, 400):
        S, *_ = make_instance(r, p, 10 * p, seed=0)
        out[f"{r}_{p}"] = S
np.savez(sys.argv[2], **out)
"""


def _draw(threads, path):
    subprocess.run([sys.executable, "-c", CHILD, str(threads), path,
                    os.path.join(ROOT, "src")], check=True)
    return np.load(path)


def test_S_independent_of_blas_threads():
    with tempfile.TemporaryDirectory() as d:
        a = _draw(1, os.path.join(d, "t1.npz"))
        b = _draw(8, os.path.join(d, "t8.npz"))
        for k in a.files:
            diff = np.abs(a[k] - b[k]).max()
            assert diff < 1e-10, f"{k}: S differs by {diff:.2e} between 1 and 8 threads"


if __name__ == "__main__":
    test_S_independent_of_blas_threads()
    print("OK: S is identical (to 1e-10) at 1 and 8 BLAS threads")
