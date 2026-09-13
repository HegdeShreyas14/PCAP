"""
batched_admm.py — batched block-sparse Graphical Lasso solver.

Why this exists
---------------
Profiling GGLasso's block_SGL (see notes/PROGRESS.md finding 4) showed the
runtime is NOT dominated by the eigendecompositions the cubic-work model
points at. In the control config (balanced p=800, 238 blocks) the largest
single `eigh` is ~1.7% of eigh time and ~1% of wall-clock. Instead:

    Python per-block glue (loop, ix_ slicing, per-call ADMM_SGL setup,
    inverse permutation)                                    35-65% of wall
    thousands of sub-millisecond eigh calls, mostly dispatch  26-58%
    ADMM_stopping_criterion (4k-10k np.linalg.norm calls)   0.12-0.35 s cumulative

So the lever is BATCHING, not a faster eigensolver.

This module solves all blocks of the same size simultaneously as a stacked
(k, s, s) array: one batched eigh, one batched prox, one batched stopping test
per group per iteration, replacing k separate Python-level ADMM runs.

Backends
--------
NumPy and PyTorch share the same code path: every operation used here is
batched over leading dimensions in both. Pass backend='torch' with a CUDA
device for the GPU version. This is deliberate -- the research contribution is
the mapping of the decomposition onto batched execution, not the framework.

Correctness
-----------
Replicates gglasso 0.3.0's ADMM_SGL exactly:
  Omega update : phiplus, i.e. 0.5*(sqrt(d^2 + 4/rho) + d) on eigenvalues
  Theta update : OFF-DIAGONAL soft-thresholding (prox_od_1norm copies the
                 diagonal through untouched). Getting this wrong silently
                 solves a different problem.
  X update     : X += Omega - Theta
  stopping     : Boyd primal/dual residuals with the same dim factor
                 ((p^2+p)/2) and the same adaptive rho rule.

Blocks of size 1 are solved in closed form (no penalty applies to a diagonal
entry), which also avoids a degenerate 1x1 eigendecomposition per singleton --
and singletons are the majority of detected blocks in fragmented regimes.
"""

import numpy as np


# --------------------------------------------------------------------------
# backend shim: numpy and torch expose the same batched API for what we need
# --------------------------------------------------------------------------

class _NumpyBackend:
    name = "numpy"

    @staticmethod
    def asarray(x):
        return np.asarray(x, dtype=np.float64)

    @staticmethod
    def eigh(A):
        return np.linalg.eigh(A)

    @staticmethod
    def to_numpy(x):
        return x

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float64)

    @staticmethod
    def eye_batch(k, s):
        return np.broadcast_to(np.eye(s), (k, s, s)).copy()

    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    sign = staticmethod(np.sign)
    maximum = staticmethod(np.maximum)

    @staticmethod
    def norm_batch(A):
        """Frobenius norm per matrix in the batch -> (k,)"""
        return np.sqrt((A ** 2).sum(axis=(-2, -1)))

    @staticmethod
    def swapaxes(A):
        return np.swapaxes(A, -1, -2)

    @staticmethod
    def all_true(mask):
        return bool(np.all(mask))

    @staticmethod
    def synchronize():
        pass  # no device to sync


class _TorchBackend:
    name = "torch"

    def __init__(self, device="cuda", dtype=None):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.float64

    def asarray(self, x):
        return self.torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    def eigh(self, A):
        return self.torch.linalg.eigh(A)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def eye_batch(self, k, s):
        return self.torch.eye(s, dtype=self.dtype, device=self.device).expand(k, s, s).clone()

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def abs(self, x):
        return self.torch.abs(x)

    def sign(self, x):
        return self.torch.sign(x)

    def maximum(self, a, b):
        if not self.torch.is_tensor(b):
            b = self.torch.tensor(b, dtype=self.dtype, device=self.device)
        return self.torch.maximum(a, b)

    def norm_batch(self, A):
        return self.torch.sqrt((A ** 2).sum(dim=(-2, -1)))

    def swapaxes(self, A):
        return A.transpose(-1, -2)

    def all_true(self, mask):
        return bool(mask.all())

    def synchronize(self):
        self.torch.cuda.synchronize()


def get_backend(name="numpy", device="cuda"):
    if name == "numpy":
        return _NumpyBackend()
    if name == "torch":
        return _TorchBackend(device=device)
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------
# batched primitives
# --------------------------------------------------------------------------

def _phiplus_batch(bk, W, beta):
    """
    Batched prox of -beta*logdet: eigendecompose, apply
    0.5*(sqrt(d^2 + 4*beta) + d), reassemble. W is (k, s, s).

    This is the single batched eigh that replaces k separate calls.
    """
    D, Q = bk.eigh(W)                       # (k,s), (k,s,s)
    dpos = 0.5 * (bk.sqrt(D * D + 4.0 * beta) + D)
    # (Q * dpos[..., None, :]) scales columns, matching gglasso's (Q * phip) @ Q.T
    return (Q * dpos[..., None, :]) @ bk.swapaxes(Q)


def _prox_od_batch(bk, A, lam, diag_mask):
    """
    Batched OFF-DIAGONAL soft-thresholding.

    gglasso's prox_od_1norm copies the diagonal through untouched
    (`res[i,i] = A[i,i]`). Thresholding the diagonal would shrink conditional
    variances toward zero and silently solve a different problem, so the
    diagonal is restored explicitly via the mask.
    """
    out = bk.sign(A) * bk.maximum(bk.abs(A) - lam, 0.0)
    return out * (1.0 - diag_mask) + A * diag_mask


# --------------------------------------------------------------------------
# batched ADMM over a group of equally-sized blocks
# --------------------------------------------------------------------------

def batched_admm_group(S_stack, lambda1, bk, rho=1.0, max_iter=1000,
                       tol=1e-7, rtol=1e-5, update_rho=True):
    """
    Solve k independent Graphical Lasso problems of identical size s at once.

    Returns
    -------
    Theta : (k, s, s), Omega : (k, s, s), iters : int, timing : dict
        timing has keys 't_h2d' (host->device transfer of S and the initial
        Omega/Theta/X buffers), 't_compute' (the ADMM loop itself), all in
        seconds. For the numpy backend both transfer costs are ~0 since there
        is no device boundary; the columns exist so numpy and torch rows share
        a schema.

    All k problems share an iteration count: the loop runs until EVERY
    problem in the batch has converged. Converged problems are frozen via a
    mask so their values do not drift.
    """
    import time as _time

    t0 = _time.perf_counter()
    S = bk.asarray(S_stack)
    k, s, _ = S.shape

    Omega = bk.eye_batch(k, s)
    Theta = bk.eye_batch(k, s)
    X = bk.zeros((k, s, s))
    diag_mask = bk.eye_batch(k, s)
    bk.synchronize()
    t_h2d = _time.perf_counter() - t0

    t0 = _time.perf_counter()

    dim = (s ** 2 + s) / 2.0
    rho_v = bk.zeros((k,)) + rho
    active = bk.zeros((k,)) + 1.0          # 1 = still iterating
    # Iteration at which each problem FIRST converged, tracked ON-DEVICE as a
    # length-k vector (0 = not yet). Read back once at the end, never per
    # iteration -- a per-iteration host copy would add its own sync overhead
    # and pollute the timing this function exists to measure. The spread
    # between earliest and latest convergence is the straggler cost of the
    # shared iteration count (T_group = max_i T_i): if it is large, unexpected
    # timing is convergence-straggler overhead, not group-count dispatch
    # overhead. The two are otherwise easy to confuse.
    conv_iter_v = bk.zeros((k,))

    iters = 0
    for it in range(max_iter):
        iters = it + 1
        r3 = rho_v[:, None, None]

        # --- Omega update: ONE batched eigh for the whole group -----------
        # beta = 1/rho varies per problem once rho adapts, so the eigenvalue
        # transform is applied elementwise rather than via a scalar helper.
        W = Theta - X - S / r3
        Omega_prev = Omega
        D, Q = bk.eigh(W)
        beta = (1.0 / rho_v)[:, None]
        dpos = 0.5 * (bk.sqrt(D * D + 4.0 * beta) + D)
        Omega_new = (Q * dpos[..., None, :]) @ bk.swapaxes(Q)

        a3 = active[:, None, None]
        Omega = a3 * Omega_new + (1.0 - a3) * Omega

        # --- Theta update: batched off-diagonal soft-threshold ------------
        Theta_new = _prox_od_batch(bk, Omega + X, lambda1 / r3, diag_mask)
        Theta = a3 * Theta_new + (1.0 - a3) * Theta

        # --- X update ------------------------------------------------------
        X = X + a3 * (Omega - Theta)

        # --- batched stopping test (one pass, not k norm calls) -----------
        r_t = bk.norm_batch(Omega - Theta)
        s_t = rho_v * bk.norm_batch(Omega - Omega_prev)
        e_pri = dim * tol + rtol * bk.maximum(bk.norm_batch(Omega),
                                              bk.norm_batch(Theta))
        e_dual = dim * tol + rtol * rho_v * bk.norm_batch(X)

        converged = (r_t <= e_pri) & (s_t <= e_dual)
        conv_f = (converged.astype(active.dtype) if hasattr(converged, "astype")
                  else converged.to(active.dtype))
        # newly converged this iteration = active AND converged. Stamp those
        # with the current iteration number; on-device, no host sync.
        newly = active * conv_f
        conv_iter_v = conv_iter_v + newly * float(it + 1)
        active = active * (1.0 - conv_f)
        if bk.all_true(active == 0):
            break

        # --- adaptive rho, same rule as gglasso ---------------------------
        if update_rho:
            up = (r_t >= 10 * s_t)
            down = (s_t >= 10 * r_t)
            f = 1.0 + 1.0 * (up if not hasattr(up, "to") else up.to(rho_v.dtype))
            f = f - 0.5 * (down if not hasattr(down, "to") else down.to(rho_v.dtype))
            rho_new = rho_v * f
            X = X * (rho_v / rho_new)[:, None, None]
            rho_v = rho_new

    bk.synchronize()
    t_compute = _time.perf_counter() - t0

    # Straggler summary, read back once. Problems that never converged (still
    # 0 in conv_iter_v) are stamped with the final iteration count so they
    # count as slowest rather than fastest.
    ci = bk.to_numpy(conv_iter_v).astype(float)
    ci[ci == 0] = iters
    conv_min = float(ci.min())
    conv_max = float(ci.max())
    timing = {"t_h2d": t_h2d, "t_compute": t_compute,
              "conv_iter_min": conv_min, "conv_iter_max": conv_max,
              "conv_iter_spread": conv_max - conv_min}
    return Theta, Omega, iters, timing


# --------------------------------------------------------------------------
# top-level solver
# --------------------------------------------------------------------------

def batched_block_SGL(S, lambda1, backend="numpy", device="cuda",
                      max_iter=1000, tol=1e-7, rtol=1e-5, rho=1.0,
                      return_info=False):
    """
    Drop-in analogue of gglasso's block_SGL using batched execution.

    Detects connected components, groups them by size, and solves each size
    group as one batched ADMM instead of looping in Python.
    """
    from gglasso.solver.single_admm_solver import get_connected_components

    p = S.shape[0]
    bk = get_backend(backend, device)

    numC, allC = get_connected_components(S, lambda1)
    Theta_full = np.zeros((p, p))

    # Singletons: no off-diagonal penalty applies to a 1x1 block, so the
    # solution is the closed form 1/S_ii. Handling these directly avoids a
    # degenerate eigendecomposition per singleton, and singletons dominate the
    # block count in fragmented regimes.
    singles = [c for c in allC if len(c) == 1]
    for c in singles:
        i = int(np.asarray(c)[0])
        Theta_full[i, i] = 1.0 / S[i, i] if S[i, i] > 0 else 1.0

    groups = {}
    for c in allC:
        idx = np.asarray(sorted(c), dtype=int)
        if idx.size <= 1:
            continue
        groups.setdefault(idx.size, []).append(idx)

    # Batch-composition memory metrics. A batched group is a (k, s, s) tensor,
    # so its footprint scales as O(k * s^2), NOT as O(p^2). Peak VRAM is driven
    # by whichever group MAXIMISES k*s^2 -- which is not necessarily the group
    # with the largest s (that block is often alone, k=1) nor the group with
    # the largest k (usually tiny crumbs). Record both the max-k and the
    # max-(k*s^2) group so RQ3 can attribute memory to tensor shape rather than
    # to global problem dimension p.
    max_group_k = max((len(v) for v in groups.values()), default=0)
    mem_group_s, mem_group_k, mem_group_ks2 = 0, 0, 0
    for size, idx_list in groups.items():
        ks2 = len(idx_list) * size * size
        if ks2 > mem_group_ks2:
            mem_group_ks2 = ks2
            mem_group_s = size
            mem_group_k = len(idx_list)

    info = {"n_blocks": numC, "n_singletons": len(singles),
            "n_groups": len(groups), "group_sizes": {}, "iters": {},
            "backend": backend,
            # Explicit label so downstream analysis never confuses this with
            # the unbatched gglasso reference: above p=1600 (see RUNBOOK) the
            # unbatched block_SGL reference is dropped and this batched-numpy
            # path becomes the CPU baseline. It is validated to ~1e-14 against
            # the reference at every size tested so far, but it is a
            # different artifact and should be labeled as such in any table
            # or figure: "CPU-best (batched)", not bare "CPU".
            "cpu_baseline_label": "CPU-best (batched numpy)" if backend == "numpy" else None,
            # batch-composition memory metrics (see comment above)
            "max_group_k": max_group_k,
            "mem_group_s": mem_group_s,      # s of the memory-dominant group
            "mem_group_k": mem_group_k,      # k of the memory-dominant group
            "mem_group_ks2": mem_group_ks2,  # its k*s^2 (proportional to VRAM)
            "t_h2d": 0.0, "t_compute": 0.0, "t_d2h": 0.0,
            "max_conv_spread": 0.0}

    import time as _time
    for size, idx_list in sorted(groups.items()):
        stack = np.stack([S[np.ix_(idx, idx)] for idx in idx_list])
        Theta_b, _, iters, timing = batched_admm_group(
            stack, lambda1, bk, rho=rho, max_iter=max_iter, tol=tol, rtol=rtol)
        t0 = _time.perf_counter()
        Theta_b = bk.to_numpy(Theta_b)
        bk.synchronize()
        t_d2h = _time.perf_counter() - t0

        for j, idx in enumerate(idx_list):
            Theta_full[np.ix_(idx, idx)] = Theta_b[j]
        info["group_sizes"][size] = len(idx_list)
        info["iters"][size] = iters
        info["t_h2d"] += timing["t_h2d"]
        info["t_compute"] += timing["t_compute"]
        info["t_d2h"] += t_d2h
        info["max_conv_spread"] = max(info["max_conv_spread"],
                                      timing["conv_iter_spread"])

    sol = {"Theta": Theta_full}
    return (sol, info) if return_info else sol