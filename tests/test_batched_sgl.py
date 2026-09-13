import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batched_sgl import batched_block_sgl
from blockgen import REGIMES, make_instance, regime_block_sizes
from gglasso.solver.single_admm_solver import block_SGL


class BlockGeneratorTests(unittest.TestCase):
    def test_regimes_partition_requested_dimension(self):
        for regime in REGIMES:
            for p in (17, 100, 203):
                sizes = regime_block_sizes(regime, p)
                self.assertEqual(sum(sizes), p)
                self.assertTrue(all(s > 0 for s in sizes))

    def test_sample_covariance_is_deterministic(self):
        first = make_instance("balanced", 40, 200, seed=7)[0]
        second = make_instance("balanced", 40, 200, seed=7)[0]
        np.testing.assert_array_equal(first, second)


class BatchedSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        S, *_ = make_instance("many_tiny", 100, 1000, seed=1)
        cls.S = S[0] if S.ndim == 3 else S
        cls.kwargs = dict(tol=1e-7, rtol=1e-5, max_iter=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.reference = block_SGL(cls.S, 0.15, np.eye(100),
                                      verbose=False, measure=False,
                                      **cls.kwargs)

    def assert_matches_reference(self, backend, dtype="float64", atol=1e-8):
        result, info = batched_block_sgl(
            self.S, 0.15, np.eye(100), backend=backend, dtype=dtype,
            batch_size=64, padding_ratio=1.5, **self.kwargs)
        self.assertTrue(info["all_converged"])
        self.assertGreaterEqual(max(info["batch_counts"]), 2)
        np.testing.assert_allclose(result["Theta"], self.reference["Theta"],
                                   rtol=atol, atol=atol)
        np.testing.assert_array_equal(
            np.abs(result["Theta"]) > 1e-6,
            np.abs(self.reference["Theta"]) > 1e-6)

    def test_numpy_matches_gglasso(self):
        self.assert_matches_reference("numpy")

    def test_torch_cpu_matches_gglasso(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        self.assert_matches_reference("torch-cpu")

    def test_cuda_matches_gglasso(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        self.assert_matches_reference("cuda", atol=2e-8)

    def test_all_singletons_closed_form(self):
        S = np.diag([2.0, 4.0, 5.0])
        result, info = batched_block_sgl(S, 0.1, backend="numpy")
        np.testing.assert_allclose(np.diag(result["Theta"]), [0.5, 0.25, 0.2])
        self.assertEqual(info["n_singletons"], 3)
        self.assertEqual(info["n_batches"], 0)


if __name__ == "__main__":
    unittest.main()
