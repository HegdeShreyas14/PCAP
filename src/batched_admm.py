"""batched_admm.py — batched block-sparse Graphical Lasso solver."""

import numpy as np


# --- backend shim ---

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


class _NumpyBackend32(_NumpyBackend):
    """float32 CPU variant: only for a like-for-like comparison with the
    float32 GPU path. Never the default baseline.
    """
    name = "numpy"

    @staticmethod
    def asarray(x):
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def eye_batch(k, s):
        return np.broadcast_to(np.eye(s, dtype=np.float32), (k, s, s)).copy()


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


def get_backend(name="numpy", device="cuda", dtype="float64"):
    """float64 is the reference and the default for both backends. float32
    exists to measure how much of the GPU result is the cost of FP64 on a
    consumer GPU, and on the CPU only so that comparison is like-for-like.
    """
    if dtype not in ("float64", "float32"):
        raise ValueError(f"unknown dtype {dtype!r}")
    if name == "numpy":
        # float64 is the CPU baseline; float32 must be requested explicitly and
        # exists only to make float32 GPU comparisons like-for-like
        return _NumpyBackend() if dtype == "float64" else _NumpyBackend32()
    if name == "torch":
        import torch
        return _TorchBackend(device=device, dtype=getattr(torch, dtype))
    raise ValueError(f"unknown backend {name!r}")



def _phiplus_batch(bk, W, beta):
    """Batched prox of -beta*logdet: eigendecompose, apply 0.5*(sqrt(d^2 +
    4*beta) + d), reassemble. W is (k, s, s).
    """
    D, Q = bk.eigh(W)                       # (k,s), (k,s,s)
    dpos = 0.5 * (bk.sqrt(D * D + 4.0 * beta) + D)
    # (Q * dpos[..., None, :]) scales columns, matching gglasso's (Q * phip) @ Q.T
    return (Q * dpos[..., None, :]) @ bk.swapaxes(Q)


def _prox_od_batch(bk, A, lam, diag_mask):
    """Batched OFF-DIAGONAL soft-thresholding."""
    out = bk.sign(A) * bk.maximum(bk.abs(A) - lam, 0.0)
    return out * (1.0 - diag_mask) + A * diag_mask



def batched_admm_group(S_stack, lambda1, bk, rho=1.0, max_iter=1000,
                       tol=1e-7, rtol=1e-5, update_rho=True):
    """Solve k independent Graphical Lasso problems of identical size s at
    once.
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
    # Iteration at which each problem first converged (0 = not yet), kept on device.
    conv_iter_v = bk.zeros((k,))

    iters = 0
    for it in range(max_iter):
        iters = it + 1
        r3 = rho_v[:, None, None]

        # --- Omega update: one batched eigh for the whole group ----------
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

    # Read convergence iterations back once; unconverged problems count as max_iter.
    ci = bk.to_numpy(conv_iter_v).astype(float)
    n_not_converged = int((ci == 0).sum())
    ci[ci == 0] = iters
    conv_min = float(ci.min())
    conv_max = float(ci.max())
    timing = {"t_h2d": t_h2d, "t_compute": t_compute,
              "conv_iter_min": conv_min, "conv_iter_max": conv_max,
              "conv_iter_spread": conv_max - conv_min,
              # problems that hit max_iter without meeting the stopping test.
              # Must be 0 for a result to count as converged.
              "n_not_converged": n_not_converged}
    return Theta, Omega, iters, timing



def batched_block_SGL(S, lambda1, backend="numpy", device="cuda",
                      max_iter=1000, tol=1e-7, rtol=1e-5, rho=1.0,
                      return_info=False, dtype="float64"):
    """Drop-in analogue of gglasso's block_SGL using batched execution."""
    from gglasso.solver.single_admm_solver import get_connected_components

    p = S.shape[0]
    bk = get_backend(backend, device, dtype)

    numC, allC = get_connected_components(S, lambda1)
    Theta_full = np.zeros((p, p))

    # Singletons have the closed-form solution 1/S_ii.
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

    # Memory scales with k*s^2 of the largest batched group, not with p.
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
            "backend": backend, "dtype": dtype,
            "n_not_converged": 0, "max_iters": 0,
            # Label so tables never confuse this with the unbatched gglasso reference.
            "cpu_baseline_label": ("CPU-best (batched numpy)" if dtype == "float64"
                                   else "CPU batched numpy float32") if backend == "numpy" else None,
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
        info["n_not_converged"] += timing["n_not_converged"]
        info["max_iters"] = max(info["max_iters"], iters)

    sol = {"Theta": Theta_full}
    return (sol, info) if return_info else sol