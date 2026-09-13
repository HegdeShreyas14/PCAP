"""Batched block-sparse graphical lasso for NumPy CPUs and PyTorch devices.

The connected-component rule is identical to :func:`gglasso.block_SGL`.
Non-singleton components of similar sizes are padded, stacked, and advanced
together through ADMM.  Every component keeps its own convergence decision
and adaptive ``rho`` value, so batching does not couple the optimization
problems.  Padding is excluded from residuals and discarded on assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np
from gglasso.solver.single_admm_solver import get_connected_components


@dataclass
class _Batch:
    components: list[np.ndarray]
    width: int


def _make_batches(components, batch_size=64, padding_ratio=1.5):
    """Group similarly-sized components while bounding padding overhead."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if padding_ratio < 1:
        raise ValueError("padding_ratio must be at least 1")
    ordered = sorted((np.asarray(c, dtype=int) for c in components),
                     key=len)
    out = []
    current = []
    smallest = None
    for comp in ordered:
        size = len(comp)
        if current and (len(current) >= batch_size or
                        size > smallest * padding_ratio):
            out.append(_Batch(current, max(map(len, current))))
            current, smallest = [], None
        if not current:
            smallest = size
        current.append(comp)
    if current:
        out.append(_Batch(current, max(map(len, current))))
    return out


def _pack(batch, S, Omega_0, Theta_0, X_0, lambda1_mask):
    b, m = len(batch.components), batch.width
    packed_s = np.zeros((b, m, m), dtype=np.float64)
    omega = np.zeros_like(packed_s)
    theta = np.zeros_like(packed_s)
    dual = np.zeros_like(packed_s)
    penalty = np.zeros_like(packed_s)
    real_mask = np.zeros_like(packed_s, dtype=bool)
    sizes = np.asarray([len(c) for c in batch.components], dtype=int)
    for j, comp in enumerate(batch.components):
        n = len(comp)
        ix = np.ix_(comp, comp)
        packed_s[j, :n, :n] = S[ix]
        omega[j, :n, :n] = Omega_0[ix]
        theta[j, :n, :n] = Theta_0[ix]
        dual[j, :n, :n] = X_0[ix]
        penalty[j, :n, :n] = lambda1_mask[ix]
        real_mask[j, :n, :n] = True
        # Independent, well-conditioned dummy variables make padding neutral.
        if n < m:
            pad = np.arange(n, m)
            packed_s[j, pad, pad] = 1.0
            omega[j, pad, pad] = 1.0
            theta[j, pad, pad] = 1.0
    return packed_s, omega, theta, dual, penalty, real_mask, sizes


def _solve_group_numpy(S, omega, theta, dual, penalty, real_mask, sizes,
                       lambda1, rho, max_iter, tol, rtol, update_rho):
    b, m, _ = S.shape
    rho_v = np.full(b, float(rho))
    iterations = np.full(b, max_iter, dtype=int)
    converged = np.zeros(b, dtype=bool)
    active = np.arange(b)
    diagonal = np.arange(m)

    for iteration in range(max_iter):
        if active.size == 0:
            break
        a = active
        old = omega[a].copy()
        rv = rho_v[a]
        W = theta[a] - dual[a] - S[a] / rv[:, None, None]
        eigvals, eigvecs = np.linalg.eigh(W)
        phi = 0.5 * (eigvals + np.sqrt(eigvals * eigvals +
                                       4.0 / rv[:, None]))
        new_omega = (eigvecs * phi[:, None, :]) @ np.swapaxes(eigvecs, -1, -2)
        A = new_omega + dual[a]
        threshold = lambda1 * penalty[a] / rv[:, None, None]
        new_theta = np.sign(A) * np.maximum(np.abs(A) - threshold, 0.0)
        new_theta[:, diagonal, diagonal] = A[:, diagonal, diagonal]
        new_dual = dual[a] + new_omega - new_theta

        mask = real_mask[a]
        def norm(x):
            return np.sqrt(np.sum(np.where(mask, x, 0.0) ** 2, axis=(1, 2)))

        r = norm(new_omega - new_theta)
        s = rv * norm(new_omega - old)
        dim = (sizes[a] ** 2 + sizes[a]) / 2.0
        e_pri = dim * tol + rtol * np.maximum(norm(new_omega), norm(new_theta))
        e_dual = dim * tol + rtol * rv * norm(new_dual)
        done = (r <= e_pri) & (s <= e_dual)

        if update_rho:
            new_rho = rv.copy()
            new_rho[r >= 10.0 * s] *= 2.0
            new_rho[s >= 10.0 * r] *= 0.5
            new_dual *= (rv / new_rho)[:, None, None]
            rho_v[a] = new_rho

        omega[a], theta[a], dual[a] = new_omega, new_theta, new_dual
        if np.any(done):
            finished = a[done]
            converged[finished] = True
            iterations[finished] = iteration + 1
        active = a[~done]
    return omega, theta, dual, iterations, converged


def _solve_group_torch(S, omega, theta, dual, penalty, real_mask, sizes,
                       lambda1, rho, max_iter, tol, rtol, update_rho,
                       device, dtype):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for a torch backend") from exc

    tdtype = torch.float64 if dtype == "float64" else torch.float32
    S = torch.as_tensor(S, dtype=tdtype, device=device)
    omega = torch.as_tensor(omega, dtype=tdtype, device=device)
    theta = torch.as_tensor(theta, dtype=tdtype, device=device)
    dual = torch.as_tensor(dual, dtype=tdtype, device=device)
    penalty = torch.as_tensor(penalty, dtype=tdtype, device=device)
    real_mask = torch.as_tensor(real_mask, dtype=torch.bool, device=device)
    sizes_t = torch.as_tensor(sizes, dtype=tdtype, device=device)
    b, m, _ = S.shape
    rho_v = torch.full((b,), float(rho), dtype=tdtype, device=device)
    iterations = torch.full((b,), max_iter, dtype=torch.int64, device=device)
    converged = torch.zeros(b, dtype=torch.bool, device=device)
    active = torch.arange(b, device=device)
    diagonal = torch.arange(m, device=device)

    for iteration in range(max_iter):
        if active.numel() == 0:
            break
        a = active
        old = omega[a].clone()
        rv = rho_v[a]
        W = theta[a] - dual[a] - S[a] / rv[:, None, None]
        eigvals, eigvecs = torch.linalg.eigh(W)
        phi = 0.5 * (eigvals + torch.sqrt(eigvals.square() +
                                          4.0 / rv[:, None]))
        new_omega = (eigvecs * phi[:, None, :]) @ eigvecs.transpose(-1, -2)
        A = new_omega + dual[a]
        threshold = lambda1 * penalty[a] / rv[:, None, None]
        new_theta = torch.sign(A) * torch.clamp(torch.abs(A) - threshold, min=0)
        new_theta[:, diagonal, diagonal] = A[:, diagonal, diagonal]
        new_dual = dual[a] + new_omega - new_theta

        mask = real_mask[a]
        def norm(x):
            return torch.sqrt(torch.sum(torch.where(mask, x, torch.zeros_like(x)).square(),
                                        dim=(1, 2)))

        r = norm(new_omega - new_theta)
        s = rv * norm(new_omega - old)
        dim = (sizes_t[a].square() + sizes_t[a]) / 2.0
        e_pri = dim * tol + rtol * torch.maximum(norm(new_omega), norm(new_theta))
        e_dual = dim * tol + rtol * rv * norm(new_dual)
        done = (r <= e_pri) & (s <= e_dual)

        if update_rho:
            new_rho = rv.clone()
            new_rho = torch.where(r >= 10.0 * s, new_rho * 2.0, new_rho)
            new_rho = torch.where(s >= 10.0 * r, new_rho * 0.5, new_rho)
            new_dual = new_dual * (rv / new_rho)[:, None, None]
            rho_v[a] = new_rho

        omega[a], theta[a], dual[a] = new_omega, new_theta, new_dual
        if torch.any(done):
            finished = a[done]
            converged[finished] = True
            iterations[finished] = iteration + 1
        active = a[~done]

    return (omega.cpu().numpy(), theta.cpu().numpy(), dual.cpu().numpy(),
            iterations.cpu().numpy(), converged.cpu().numpy())


def batched_block_sgl(
    S: np.ndarray,
    lambda1: float,
    Omega_0: Optional[np.ndarray] = None,
    Theta_0: Optional[np.ndarray] = None,
    X_0: Optional[np.ndarray] = None,
    *,
    backend: str = "numpy",
    batch_size: int = 64,
    padding_ratio: float = 1.5,
    rho: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-7,
    rtol: float = 1e-5,
    update_rho: bool = True,
    lambda1_mask: Optional[np.ndarray] = None,
    dtype: str = "float64",
):
    """Solve block SGL using batched ADMM.

    ``backend`` may be ``numpy``, ``torch-cpu``, or ``cuda``.  The returned
    tuple is ``(solution, info)``; solution uses the same Omega/Theta/X keys as
    GGLasso and info records component-level convergence and batching.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")
    if lambda1 <= 0 or rho <= 0:
        raise ValueError("lambda1 and rho must be positive")
    if backend not in {"numpy", "torch-cpu", "cuda"}:
        raise ValueError("backend must be numpy, torch-cpu, or cuda")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")

    p = S.shape[0]
    Omega_0 = np.eye(p) if Omega_0 is None else np.asarray(Omega_0, dtype=np.float64)
    Theta_0 = Omega_0.copy() if Theta_0 is None else np.asarray(Theta_0, dtype=np.float64)
    X_0 = np.zeros((p, p)) if X_0 is None else np.asarray(X_0, dtype=np.float64)
    lambda1_mask = (np.ones((p, p)) if lambda1_mask is None
                    else np.asarray(lambda1_mask, dtype=np.float64))
    for name, value in (("Omega_0", Omega_0), ("Theta_0", Theta_0),
                        ("X_0", X_0), ("lambda1_mask", lambda1_mask)):
        if value.shape != S.shape:
            raise ValueError(f"{name} must have shape {S.shape}")

    _, components = get_connected_components(S, lambda1 * lambda1_mask)
    singletons = [c for c in components if len(c) == 1]
    batches = _make_batches([c for c in components if len(c) > 1],
                            batch_size=batch_size, padding_ratio=padding_ratio)
    solution = {key: np.zeros((p, p), dtype=np.float64)
                for key in ("Omega", "Theta", "X")}
    all_iterations, all_converged = [], []

    for comp in singletons:
        j = int(comp[0])
        value = 1.0 / S[j, j]
        solution["Omega"][j, j] = value
        solution["Theta"][j, j] = value
        all_iterations.append(0)
        all_converged.append(True)

    started = time.perf_counter()
    device = None
    if backend != "numpy":
        import torch
        device = "cuda" if backend == "cuda" else "cpu"
        if backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")

    for batch in batches:
        packed = _pack(batch, S, Omega_0, Theta_0, X_0, lambda1_mask)
        if backend == "numpy":
            solved = _solve_group_numpy(*packed, lambda1, rho, max_iter,
                                        tol, rtol, update_rho)
        else:
            solved = _solve_group_torch(*packed, lambda1, rho, max_iter,
                                        tol, rtol, update_rho, device, dtype)
        omega, theta, dual, iterations, converged = solved
        for j, comp in enumerate(batch.components):
            n = len(comp)
            ix = np.ix_(comp, comp)
            solution["Omega"][ix] = omega[j, :n, :n]
            solution["Theta"][ix] = theta[j, :n, :n]
            solution["X"][ix] = dual[j, :n, :n]
        all_iterations.extend(iterations.tolist())
        all_converged.extend(converged.tolist())

    if backend == "cuda":
        import torch
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    info = {
        "backend": backend,
        "dtype": dtype,
        "runtime": elapsed,
        "n_components": len(components),
        "n_singletons": len(singletons),
        "n_batches": len(batches),
        "batch_widths": [b.width for b in batches],
        "batch_counts": [len(b.components) for b in batches],
        "iterations": all_iterations,
        "all_converged": bool(all(all_converged)),
    }
    return solution, info

