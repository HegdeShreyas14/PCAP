# Findings

## Final experiment results (2026-09-13)

The paper-grade run is complete on the primary machine:

- Corrected RQ1: 384 paired configurations at 1 BLAS thread and 384 at
  8 BLAS threads, each with nine trials.
- Percolation: 3,600 records covering four regimes, four dimensions, three
  sample ratios, three seeds, and 25 lambda values.
- Phase 3: 144 paired configurations and 24 best-F1 operating points.
- All six tests pass, including CUDA equivalence on the RTX 4050 Laptop GPU.

At each configuration's best-F1 lambda, median end-to-end speedup versus full
8-thread ADMM was 3.91x for GGLasso `block_SGL`, 4.32x for batched NumPy,
2.34x for batched PyTorch CPU, and 0.50x for batched CUDA. CUDA was faster than
full ADMM in 8/24 operating points: all were p=400 cases. Its maximum was 3.31x
(balanced, p=400, n/p=2), so the final result is nuanced rather than simply
negative: GPU overhead dominates at p<=200, while useful acceleration appears
at p=400.

All float64 batched backends comfortably pass the numerical acceptance gate.
Across all 144 configurations, the worst maximum absolute difference from
`block_SGL` was 6.46e-14 and the worst objective relative gap was 7.36e-16
(limits: 1e-6 and 1e-8 respectively).

The multi-seed percolation study found clear inter-block headroom at only 3 of
48 regime/dimension/sample-ratio groups, with 4 threshold ties, 4 groups that
never percolated on the tested grid, and 37 where the best-F1 lambda remained
below the percolation threshold. This confirms the structural conclusion:
finite-sample component detection merges true blocks at statistically useful
lambda even when the oracle block structure has substantial parallelism.

The corrected threading comparison is not monotone. Across all shared lambda
values, median 1-thread -> 8-thread block speedups changed from 4.04x -> 7.66x
(balanced), 6.20x -> 9.84x (imbalanced), 2.37x -> 2.15x (many-tiny), and
7.82x -> 10.26x (one-giant). At larger p the retained fraction often falls
below 100%, so threading effects must be reported per operating point with the
stored trial bands rather than summarized as a universal penalty or benefit.

## Earlier Phase-3 smoke validation (superseded by final results)

- Batched ADMM is implemented for NumPy CPU, PyTorch CPU, and CUDA.
- Six automated tests pass, including CUDA-to-GGLasso numerical equivalence.
- On the 200-variable smoke benchmark at the best-F1 lambda (0.15), batched
  NumPy achieved 3.95x versus full ADMM for the balanced instance and 11.91x
  for many-tiny. The original `block_SGL` achieved 4.00x and 7.16x.
- End-to-end CUDA was slower at these small detected block sizes: 0.49x and
  0.79x versus full ADMM. This includes host/device transfers and assembly.
- These three-trial numbers are retained only as an implementation history;
  use the nine-trial final results above for every claim and figure.
- At p=100, balanced and many-tiny both generate `[20] * 5`; the regimes are
  indistinguishable at that dimension. Phase-3 smoke testing therefore uses
  p=200.

## Interpretation guardrails

- A negative CUDA result is valid if equivalence tests pass; GPU speedup is not
  assumed in advance.
- Batched NumPy isolates the value of batching from the value of hardware.
- Final claims use end-to-end solver time and a lambda chosen only by F1.
