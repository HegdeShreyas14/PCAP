"""
blockgen.py — generate block-sparse ground-truth precision matrices with
ARBITRARY block-size distributions.

Why this exists
---------------
GGLasso's generate_precision_matrix(p, M) asserts M*L == p, so it can only
produce M blocks of EQUAL size p/M. Our methodology requires four regimes:

    balanced     [100,100,100,100,100]
    imbalanced   [450,20,20,5,5]
    many tiny    [20]*25
    one giant    [500]

Only the first is reachable with the stock generator. This module builds a
block-diagonal precision matrix by generating each block independently with
GGLasso's own generator and assembling them, so the ground-truth block
structure is exactly what we specify.
"""

import numpy as np
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix


def generate_block_precision(block_sizes, seed=None, style="powerlaw",
                             strict=False, retries=12):
    """
    Build a block-diagonal precision matrix with the given block sizes.

    Parameters
    ----------
    block_sizes : list[int]
        Size of each independent block. Blocks of size 1 and 2 are handled
        specially (see below). Total p = sum(block_sizes).
    seed : int or None
        Base seed; each block uses seed+i so blocks differ but the whole
        instance is reproducible.
    style : str
        'powerlaw' or 'erdos', passed through to GGLasso.
    strict : bool
        If True, raise instead of falling back to Erdos-Renyi when powerlaw
        generation fails. Use this when you need a guarantee that every block
        came from the requested model.
    retries : int
        Attempts at powerlaw generation before falling back.

    Returns
    -------
    Sigma : (p, p) covariance matrix
    Theta : (p, p) block-diagonal precision matrix (the ground truth)
    true_blocks : list[np.ndarray]
        Index arrays for each true block, for scoring block recovery later.
    styles : list[str]
        Which model actually produced each block: 'powerlaw', 'erdos', or
        'direct' (for size 1-2 blocks built analytically).

        This is returned, not just logged, because the fallback is NOT
        machine-independent. networkx's random_powerlaw_tree can fail to find
        a valid degree sequence, and whether it does depends on the networkx
        version and the seed: on networkx 3.6.1 a 200-node block succeeds on
        the first try, while other environments have been observed to fail on
        the same block and fall back to Erdos-Renyi. Silently falling back
        would mean two machines running identical code generate data from
        different models -- which would invalidate any cross-machine
        comparison. The caller must record this.
    """
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

    # Guarantee positive definiteness of the assembled matrix. Each block is
    # PD by construction, but assembling and rounding can leave the smallest
    # eigenvalue marginal; nudge the diagonal if so.
    min_eig = np.linalg.eigvalsh(Theta).min()
    if min_eig < 1e-6:
        Theta += (1e-6 - min_eig + 1e-3) * np.eye(p)

    # inv, not pinv: Theta is PD by construction after the nudge above, so a
    # plain inverse is correct and faster. pinv (SVD-based) would silently
    # succeed on a singular Theta and hide a real bug.
    Sigma = np.linalg.inv(Theta)
    # symmetrize against round-off
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma, Theta, true_blocks, styles


# --- the four regimes from the methodology, parameterized by total p --------

def regime_block_sizes(regime, p):
    """
    Return a block-size list for the named regime at total size p.
    Sizes are scaled from the p=500 reference used in the methodology.
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
    """
    Full pipeline: ground-truth block structure -> Sigma -> sampled S.

    Returns (S, Theta_true, true_blocks, block_sizes, styles).

    `styles` says which network model actually produced each block. Record it:
    powerlaw generation can fail and fall back to Erdos-Renyi in a way that
    varies by networkx version, so two machines running identical code are not
    guaranteed to draw from the same model unless this is checked.
    """
    sizes = regime_block_sizes(regime, p)
    Sigma, Theta, true_blocks, styles = generate_block_precision(
        sizes, seed=seed, style=style, strict=strict)
    # The sample draw MUST be seeded. Without it, Theta is reproducible but S
    # is a fresh draw on every call, so two runs of the same configuration --
    # or two lambda points within one sweep -- silently use different data.
    # This matters most for F_max: near the percolation threshold, whether the
    # giant component survives is close to a coin flip per draw. Measured at
    # balanced p=400, lambda=0.15, six draws gave F_max of 97%, 36%, 85%, 37%,
    # 37%, 97% while F1 stayed within 0.011. Structural quantities are far more
    # instance-sensitive than statistical ones.
    #
    # Offset so the sample seed differs from the network seed; using the same
    # integer for both would correlate the graph and the draw.
    S, _ = sample_covariance_matrix(Sigma, n_samples,
                                    seed=None if seed is None else seed + 90000)
    return S, Theta, true_blocks, sizes, styles


if __name__ == "__main__":
    import collections
    for r in REGIMES:
        sizes = regime_block_sizes(r, 500)
        _, _, _, styles = generate_block_precision(sizes, seed=0)
        counts = collections.Counter(styles)
        print(f"{r:12s} n_blocks={len(sizes):3d} sizes={sizes[:8]}"
              f"{'...' if len(sizes) > 8 else ''} sum={sum(sizes)} "
              f"models={dict(counts)}")