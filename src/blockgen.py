"""blockgen.py — generate block-sparse ground-truth precision matrices with
arbitrary block-size distributions.
"""

import numpy as np
from gglasso.helper.data_generation import generate_precision_matrix


def generate_block_precision(block_sizes, seed=None, style="powerlaw",
                             strict=False, retries=12):
    """Build a block-diagonal precision matrix with the given block sizes."""
    p = int(sum(block_sizes))
    Theta = np.zeros((p, p))
    true_blocks = []
    styles = []

    offset = 0
    for i, s in enumerate(block_sizes):
        s = int(s)
        blk_seed = None if seed is None else seed + i

        if s <= 2:
            # GGLasso's network generator needs a few nodes to build a graph.
            # For size 1 or 2 blocks, construct a valid PD block directly.
            B = np.eye(s)
            if s == 2:
                rng = np.random.default_rng(blk_seed)
                off = rng.uniform(0.3, 0.6) * rng.choice([-1.0, 1.0])
                B = np.array([[1.0, off], [off, 1.0]])
            used = "direct"
        else:
            B, used = None, None
            for attempt in range(retries):
                try:
                    trial_seed = None if blk_seed is None else blk_seed + 1000 * attempt
                    _, B = generate_precision_matrix(p=s, M=1, style=style,
                                                     seed=trial_seed)
                    used = style
                    break
                except Exception:
                    continue
            if B is None:
                if strict:
                    raise RuntimeError(
                        f"powerlaw generation failed for a block of size {s} "
                        f"after {retries} attempts, and strict=True. Either "
                        f"raise `retries`, or pass style='erdos' explicitly so "
                        f"every machine uses the same model."
                    )
                _, B = generate_precision_matrix(p=s, M=1, style="erdos",
                                                 prob=min(0.3, 4.0 / s),
                                                 seed=blk_seed)
                used = "erdos"

        Theta[offset:offset + s, offset:offset + s] = B
        true_blocks.append(np.arange(offset, offset + s))
        styles.append(used)
        offset += s

    # Guarantee positive definiteness of the assembled matrix.
    min_eig = np.linalg.eigvalsh(Theta).min()
    if min_eig < 1e-6:
        Theta += (1e-6 - min_eig + 1e-3) * np.eye(p)

    # inv, not pinv: Theta is PD here, and pinv would hide a singular matrix.
    Sigma = np.linalg.inv(Theta)
    # symmetrize against round-off
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma, Theta, true_blocks, styles


# --- the four regimes from the methodology, parameterized by total p --------

def regime_block_sizes(regime, p):
    """Return a block-size list for the named regime at total size p. Sizes are
    scaled from the p=500 reference used in the methodology.
    """
    if regime == "balanced":
        # 5 equal blocks
        base = p // 5
        sizes = [base] * 5
        sizes[0] += p - sum(sizes)
        return sizes

    if regime == "imbalanced":
        # one dominant block plus a tail of small ones (450/20/20/5/5 at p=500)
        sizes = [round(p * f) for f in (0.90, 0.04, 0.04, 0.01, 0.01)]
        sizes = [max(1, s) for s in sizes]
        sizes[0] += p - sum(sizes)
        return sizes

    if regime == "many_tiny":
        # blocks of ~20, as many as fit
        k = max(2, p // 20)
        base = p // k
        sizes = [base] * k
        sizes[0] += p - sum(sizes)
        return sizes

    if regime == "one_giant":
        return [p]

    raise ValueError(f"unknown regime: {regime}")


REGIMES = ["balanced", "imbalanced", "many_tiny", "one_giant"]


def make_instance(regime, p, n_samples, seed=0, style="powerlaw", strict=False):
    """Full pipeline: ground-truth block structure -> Sigma -> sampled S."""
    sizes = regime_block_sizes(regime, p)
    Sigma, Theta, true_blocks, styles = generate_block_precision(
        sizes, seed=seed, style=style, strict=strict)
    # Seed the sample draw so every run of a configuration sees the same S.
    S = sample_covariance(Sigma, n_samples,
                          seed=None if seed is None else seed + 90000)
    return S, Theta, true_blocks, sizes, styles


def sample_covariance(Sigma, n_samples, seed=None):
    """Draw n_samples from N(0, Sigma) and return the biased sample covariance."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n_samples, Sigma.shape[0])) @ L.T
    # same estimator as gglasso: np.cov(sample, bias=True), i.e. mean-centred, /N
    return np.cov(X, rowvar=False, bias=True)


if __name__ == "__main__":
    import collections
    for r in REGIMES:
        sizes = regime_block_sizes(r, 500)
        _, _, _, styles = generate_block_precision(sizes, seed=0)
        counts = collections.Counter(styles)
        print(f"{r:12s} n_blocks={len(sizes):3d} sizes={sizes[:8]}"
              f"{'...' if len(sizes) > 8 else ''} sum={sum(sizes)} "
              f"models={dict(counts)}")