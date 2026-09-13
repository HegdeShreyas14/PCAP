# Final Results and Publication-Readiness Assessment

**Project:** GPU-Accelerated Sparse Graphical Lasso  
**Final experiment date:** 13 September 2026  
**Primary hardware:** AMD64 Family 25 Model 124 CPU (12 logical cores), NVIDIA RTX 4050 Laptop GPU (6 GB)  
**Software path:** GGLasso 0.3.0 baseline; NumPy, PyTorch CPU, and CUDA batched implementations

## 1. Executive conclusion

This project has produced a real and defensible research result, not merely a
software implementation. Its central finding is:

> Batched block-sparse graphical lasso is useful, but GPU acceleration is not
> automatic. On this consumer GPU, end-to-end CUDA execution is slower for
> p <= 200, becomes beneficial in selected p=400 cases, and is constrained by
> whether statistically useful regularization reveals independent blocks at
> all.

The result is strengthened by two related findings:

1. Batched execution itself is valuable. At the best-support-F1 operating
   points, batched NumPy has a median 4.32x speedup over full 8-thread ADMM,
   compared with 3.91x for the original block-by-block GGLasso implementation.
2. The primary obstacle to block-parallel acceleration is partly statistical,
   not merely computational. In 37 of 48 percolation groups, the lambda that
   maximizes support recovery is below the threshold needed to break the
   detected graph into useful independent components.

### Is it publishable?

**Yes, with scope-dependent qualifications.** In its current form this is:

- publishable as a strong undergraduate/masters project report;
- a credible candidate for a student research symposium, poster, reproducibility
  track, systems-for-ML workshop, or short-paper venue;
- not yet sufficiently validated for a strong full-length archival conference
  or journal paper.

Before an external paper submission, the most important additions are
multi-seed Phase-3 benchmarks, at least one real dataset, stronger baseline
coverage, and confidence intervals across independent generated instances.
Those additions are about external validity, not about repairing the current
implementation: the present numerical and internal-validity checks are strong.

## 2. Research problem

The single graphical lasso estimates a sparse precision matrix Theta by
minimizing

```text
-log det(Theta) + trace(S Theta) + lambda ||Theta||_1,off
```

where S is the empirical covariance matrix and the L1 penalty applies only to
off-diagonal entries. The usual ADMM solver repeatedly performs an
eigendecomposition of a p by p matrix, which is expensive as p grows.

For a fixed lambda, thresholding S can reveal connected components. If the
variables separate into independent components, the full problem can be
solved as several smaller graphical-lasso problems and reassembled exactly.
GGLasso already implements this decomposition through `block_SGL`, but it
solves the detected blocks one at a time in Python.

The project asked three connected questions:

1. When does connected-component decomposition accelerate graphical lasso
   without destroying support-recovery quality?
2. Does the detected component structure provide enough independent work for
   parallel execution?
3. Can batching those independent ADMM problems make a consumer GPU faster
   than a fair multithreaded CPU baseline?

## 3. What was implemented

The work now includes:

- deterministic generation of block-sparse precision matrices in four regimes;
- corrected paired CPU sweeps at one and eight BLAS threads;
- multi-seed percolation analysis over a dense lambda grid;
- profiling that separates eigendecomposition, component detection, assembly,
  and Python orchestration cost;
- an independent batched ADMM implementation for NumPy, PyTorch CPU, and CUDA;
- grouping and padding of similarly sized components without coupling their
  optimization states;
- independent convergence and adaptive-rho behavior for every block;
- end-to-end timing, including GPU transfer and solution reassembly;
- numerical-equivalence tests against GGLasso;
- resumable JSONL checkpoints for every long experiment;
- automated best-F1 result selection and final figure generation.

The batched solver is not an approximation to the block solution. In float64,
it solves the same independent optimization problems while changing how the
linear algebra and stopping checks are scheduled.

## 4. Experimental design

### 4.1 Synthetic structural regimes

Four regimes expose different component-size distributions:

| Regime | Purpose |
|---|---|
| Balanced | Several similarly sized blocks; favorable to batching |
| Imbalanced | Mixed component sizes; tests grouping and padding overhead |
| Many-tiny | Numerous small blocks; stresses Python/kernel-launch overhead |
| One-giant | One dominant component plus smaller blocks; adverse control |

These regimes are necessary because reporting only a balanced structure would
systematically favor block-parallel methods.

### 4.2 Corrected RQ1 CPU sweep

- 4 regimes
- p in {100, 200, 400, 800}
- sample ratio n/p in {2, 5, 10}
- 8 lambda values
- 384 configurations per threading condition
- 9 timing trials per solver
- identical seed and generated instance in the 1-thread and 8-thread files

This sweep compares full `ADMM_SGL` with GGLasso `block_SGL`. The final audit
confirmed that the two CSVs have exactly the same 384 configuration keys.

### 4.3 Percolation study

- 4 regimes
- p in {100, 200, 400, 800}
- n/p in {2, 5, 10}
- seeds {0, 1, 2}
- 25 lambda values between 0.02 and 0.50
- 3,600 final rows

For each lambda, the study records the detected number of components, largest
component, support F1, oracle block structure, and cubic-work distribution.

### 4.4 Final Phase-3 comparison

- 4 regimes
- p in {100, 200, 400}
- n/p in {2, 10}
- 6 lambda values
- 144 paired configurations
- 9 measured trials after 2 warm-up runs
- one fixed generated instance per configuration (`seed=0`)
- 8 BLAS threads for the CPU baseline
- float64 on NumPy, PyTorch CPU, and CUDA

The compared execution paths are:

1. full-matrix GGLasso ADMM;
2. sequential GGLasso `block_SGL`;
3. batched NumPy CPU;
4. batched PyTorch CPU;
5. batched PyTorch CUDA.

Performance operating points are chosen solely by maximum support F1, never by
runtime. This prevents an artificially large speedup from being reported at a
lambda that has fragmented the graph by deleting real edges.

## 5. Metrics and how to read them

### Support F1

Support F1 compares the nonzero off-diagonal pattern of the estimate with the
known synthetic ground truth. It balances precision and recall. It is used to
select the statistically defensible lambda.

### End-to-end speedup

```text
speedup = median time of full ADMM / median time of tested solver
```

A value above 1 means the tested solver is faster. CUDA timing includes device
transfer, batched optimization, synchronization, and final matrix assembly.

### Numerical equivalence

Two checks compare each batched result with `block_SGL`:

- maximum absolute elementwise difference;
- relative gap in the graphical-lasso objective.

The acceptance limits were 1e-6 and 1e-8 respectively.

### Percolation and cubic-work fraction

For detected component sizes s_i, the cubic-work proxy is based on sum(s_i^3).
`F_max` is the fraction of this proxy assigned to the largest detected block.
The percolation threshold is the smallest lambda where `F_max < 0.90`.

The proxy is a structural diagnostic, not a wall-clock predictor. Profiling
showed that it overweights large blocks because it ignores per-block iteration
counts and thousands of small Python and linear-algebra dispatches.

## 6. Correctness and reproducibility results

### 6.1 Numerical acceptance

Across all 144 Phase-3 configurations:

| Backend | Worst max absolute difference | Worst objective relative gap |
|---|---:|---:|
| Batched NumPy | 4.00e-14 | 4.75e-16 |
| Batched PyTorch CPU | 6.35e-14 | 5.70e-16 |
| Batched CUDA | 6.45e-14 | 7.36e-16 |
| Acceptance limit | 1.00e-6 | 1.00e-8 |

Every backend is many orders of magnitude inside the acceptance limits. The
performance comparison is therefore between numerically equivalent solutions,
not between a correct CPU solver and a lower-accuracy GPU approximation.

### 6.2 Automated tests

All six tests pass:

- all-singleton closed-form behavior;
- NumPy equivalence with GGLasso;
- PyTorch CPU equivalence with GGLasso;
- CUDA equivalence with GGLasso;
- correct dimension partitioning in every generator regime;
- deterministic empirical covariance generation.

### 6.3 Measurement integrity

The final audit also verified:

- 384 rows in each RQ1 file;
- zero key differences between the paired RQ1 grids;
- 3,600 percolation rows;
- 144 Phase-3 rows and 24 best-F1 rows;
- nine stored timings for every solver in every Phase-3 configuration;
- successful rendering of the final result figure;
- no whitespace errors in the source/documentation diff.

## 7. Phase-3 performance results

### 7.1 Overall result at best-F1 lambda

Each median below gives equal weight to the 24 regime/p/sample-ratio operating
points.

| Solver | Median speedup vs full ADMM | Faster than ADMM | Range |
|---|---:|---:|---:|
| GGLasso `block_SGL` | 3.91x | 24/24 | 1.11x-11.24x |
| Batched NumPy CPU | **4.32x** | 23/24 | 0.98x-17.22x |
| Batched PyTorch CPU | 2.34x | 24/24 | 1.06x-8.58x |
| Batched CUDA | 0.50x | 8/24 | 0.19x-3.31x |

The strongest overall implementation on this experiment is batched NumPy, not
CUDA. That is an important result rather than a failed objective: eliminating
per-block interpreter and dispatch overhead matters more than moving every
small problem to a GPU.

### 7.2 Scaling with dimension

| p | `block_SGL` | Batched NumPy | PyTorch CPU | CUDA |
|---:|---:|---:|---:|---:|
| 100 | 3.22x | 2.70x | 1.26x | 0.26x |
| 200 | 3.85x | 4.12x | 2.34x | 0.50x |
| 400 | 5.13x | 5.47x | **6.81x** | **2.35x** |

This table contains the clearest GPU crossover result. CUDA is decisively poor
at p=100 and p=200, but its median reaches 2.35x at p=400. PyTorch CPU also
becomes the strongest median backend at p=400, suggesting that tensorized
batching begins to amortize its framework overhead at this scale.

### 7.3 CUDA wins

CUDA is faster than full ADMM at eight best-F1 points, all at p=400:

| Regime | n/p | Best lambda | F1 | CUDA speedup |
|---|---:|---:|---:|---:|
| Balanced | 2 | 0.20 | 0.716 | **3.31x** |
| Balanced | 10 | 0.10 | 0.871 | 2.08x |
| Imbalanced | 2 | 0.15 | 0.704 | 2.09x |
| Imbalanced | 10 | 0.10 | 0.853 | 1.67x |
| Many-tiny | 2 | 0.25 | 0.673 | 2.99x |
| Many-tiny | 10 | 0.20 | 0.792 | 3.07x |
| One-giant | 2 | 0.10 | 0.600 | 1.18x |
| One-giant | 10 | 0.05 | 0.807 | 2.62x |

The two n/p=5 cases at p=400 are not included in Phase 3 by design; Phase 3
uses n/p in {2, 10}. The eight wins therefore cover every tested p=400
regime/sample-ratio pair.

### 7.4 Results by structural regime

| Regime | `block_SGL` | Batched NumPy | PyTorch CPU | CUDA |
|---|---:|---:|---:|---:|
| Balanced | 4.29x | 4.46x | 2.11x | 0.47x |
| Imbalanced | 4.76x | **5.70x** | 2.50x | 0.55x |
| Many-tiny | 2.77x | 2.73x | **2.90x** | 0.59x |
| One-giant | 3.67x | 3.52x | 2.20x | 0.36x |

The one-giant regime produces the weakest median CUDA result, as expected:
there is less independent work to batch. The many-tiny result also shows that
having many blocks is not sufficient. If each block is extremely small,
framework and kernel-launch costs can dominate.

## 8. RQ1 CPU and threading results

Across the full shared lambda grid, median block-solver speedup changes as
follows when moving from the 1-thread baseline run to the 8-thread baseline
run:

| Regime | 1-thread file | 8-thread file | 8t/1t retained ratio |
|---|---:|---:|---:|
| Balanced | 4.04x | 7.66x | 190% |
| Imbalanced | 6.20x | 9.84x | 159% |
| Many-tiny | 2.37x | 2.15x | 91% |
| One-giant | 7.82x | 10.26x | 131% |

This does not mean eight threads universally make block decomposition look
better. At individual large-p best-F1 points, the retained ratio frequently
falls below 100%. For example, balanced p=800 retains only 58%-75%, depending
on sample ratio. At smaller p, timing noise, BLAS scheduling, and fixed overhead
can reverse the apparent direction.

The defensible conclusion is therefore:

> BLAS threading changes full and decomposed solvers differently, and the
> effect depends on dimension and component structure. Report the fair
> 8-thread result and its trial band; use the 1-thread run as an algorithmic
> comparison, not as a universal upper bound.

## 9. Percolation result

The final percolation table contains 48 regime/p/sample-ratio groups:

| Verdict | Groups | Interpretation |
|---|---:|---|
| No inter-block work | 37 | Best-F1 lambda is below the percolation threshold |
| At threshold | 4 | Best-F1 lambda and threshold coincide on the grid |
| Headroom available | 3 | Useful statistical point lies above the threshold |
| Never percolates | 4 | `F_max` never falls below 0.90 on the tested grid |

Clear headroom occurs only for balanced p=100 n/p=10, many-tiny p=100
n/p=10, and many-tiny p=200 n/p=10. The four threshold ties should not be
presented as robust headroom because the lambda grid is finite and the
threshold varies across seeds.

The oracle comparison is especially important. Balanced and many-tiny ground
truths have low oracle `F_max`, so their true block structures contain abundant
parallelism. Nevertheless, empirical covariance thresholding often merges
these blocks into a detected giant component at the lambda that best recovers
the support. This distinguishes two statements:

- Incorrect: graphical-lasso problems inherently lack block parallelism.
- Supported: finite-sample component detection often fails to expose the true
  block parallelism at the statistically preferred regularization level.

This is a publishable methodological observation because it identifies a
limit of the standard detection-plus-solve pipeline and motivates improved
component detection.

## 10. Why the GPU does not always win

The original intuition was that many independent blocks should map naturally
to a GPU. The measurements show four reasons this is incomplete.

### 10.1 Launch and framework overhead

At p<=200, the detected blocks and batched eigendecompositions are too small
to amortize PyTorch dispatch, CUDA launches, synchronization, transfer, and
assembly. The 0.26x and 0.50x CUDA medians at p=100 and p=200 quantify this.

### 10.2 Statistical regularization controls available parallelism

Increasing lambda fragments the graph and makes computation easier, but after
a point it also deletes true edges and lowers support F1. A benchmark that
selects lambda for maximum speed would therefore overstate acceleration.

### 10.3 A giant detected component serializes the useful work

Many small fragments do not help if one component holds almost all meaningful
linear-algebra work. This is common below the percolation threshold.

### 10.4 Consumer GPUs are weak at float64

The final comparison deliberately uses float64 to establish close equivalence
with the CPU reference. An RTX 4050 is optimized much more strongly for lower
precision than for float64. A float32 or mixed-precision ablation may improve
speed, but must report the resulting objective and support differences. It
would be a separate accuracy/performance result, not a replacement for the
validated float64 experiment.

## 11. What is genuinely novel

The strongest contribution is not simply “we ran graphical lasso on a GPU.”
That claim would be too broad and unlikely to be novel. The more defensible
contribution is the combination of:

1. quality-guarded performance measurement at best support F1;
2. profiling evidence that motivates batching rather than only replacing an
   eigensolver;
3. a numerically equivalent batched implementation across three backends;
4. an empirical dimension-dependent CPU/GPU crossover on consumer hardware;
5. a percolation analysis connecting statistical regularization to available
   block-level parallelism;
6. an oracle diagnostic showing that noisy detection, not necessarily the true
   graph, destroys the exploitable structure.

That combined story is substantially stronger than a raw speedup table.

## 12. Publication-readiness rubric

| Criterion | Current strength | Assessment |
|---|---|---|
| Problem importance | Good | Sparse precision estimation is widely useful and computationally expensive |
| Novelty | Moderate | Batching alone is incremental; quality/percolation/crossover analysis strengthens it |
| Correctness | Strong | Tight numerical equivalence and automated CPU/CUDA tests |
| Experimental discipline | Strong | Paired inputs, warm-ups, nine trials, full vectors, end-to-end GPU timing |
| Reproducibility | Strong | Deterministic generator, environment capture, runbook, checkpoints |
| Statistical generalization | Moderate-to-weak | Phase 3 uses one generated seed per configuration |
| External validity | Weak | Final solver benchmark is synthetic and uses one machine/GPU |
| Baseline breadth | Moderate-to-weak | Strong comparison within GGLasso, limited comparison with other solvers |
| Scale coverage | Moderate | GPU crossover appears by p=400, but larger Phase-3 dimensions are untested |
| Negative-result value | Strong | Clearly explains when and why GPU acceleration fails |
| Artifact quality | Good | Code, tests, CSVs, plot, and detailed documentation are present |

### Overall verdict

**Current readiness: strong project report; promising workshop/short paper.**

The present work is publishable if the venue accepts careful empirical studies
or student work and the claims remain limited to the tested setting. For a
competitive full paper, reviewers will likely ask whether the crossover holds
across independent graphs, real data, larger p, other GPUs, and other graphical
lasso implementations.

## 13. Required additions before external submission

### Must do

1. **Repeat Phase 3 across independent seeds.** Use at least 5 seeds if time
   permits. Report median and a 95% interval across instances, not merely nine
   repeated timings of one instance.
2. **Add at least one real dataset.** Suitable categories include gene-expression,
   financial-return, or neuroscience covariance data. Without synthetic truth,
   select lambda by a defensible criterion such as held-out likelihood or an
   information criterion rather than oracle support F1.
3. **Add baseline breadth.** Compare with at least one widely used alternative
   graphical-lasso implementation and clearly harmonize convergence tolerances.
4. **Run a percolation sensitivity analysis.** Repeat the verdict distribution
   for `F_max` thresholds such as 0.80, 0.90, and 0.95 and with a finer lambda
   grid around each transition.
5. **Report independent-instance uncertainty.** Timing bands over repeated calls
   capture machine noise; they do not capture variation caused by graph and
   sample generation.

### Strongly recommended

6. Extend Phase 3 to p=800 where memory permits.
7. Add batch-size and padding-ratio ablations.
8. Compare float64, float32, and possibly mixed precision with explicit error
   and support-quality measurements.
9. Separate transfer, detection, optimization, and assembly time while keeping
   end-to-end time as the primary metric.
10. Measure peak GPU memory and explain the largest feasible p.
11. Run on at least one second GPU or CPU architecture.
12. Verify all bibliography DOIs and format the final figure for the selected
    two-column template.

## 14. Claims that are safe to publish

The current evidence supports statements such as:

- “On an RTX 4050 Laptop GPU using float64, batched CUDA is slower than full
  ADMM at p<=200 but reaches a median 2.35x speedup across tested p=400
  best-F1 operating points.”
- “Batched NumPy provides the strongest overall median speedup, demonstrating
  that batching and reduced orchestration overhead matter independently of
  GPU hardware.”
- “Across 144 configurations, all batched backends match `block_SGL` to within
  6.46e-14 maximum absolute error.”
- “In 37 of 48 multi-seed groups, the best-F1 lambda lies below the detected
  percolation threshold, limiting inter-block parallelism at the statistically
  preferred point.”

The evidence does **not** yet support statements such as:

- “GPU graphical lasso is generally faster than CPU graphical lasso.”
- “The method generalizes to real-world covariance matrices.”
- “The crossover always occurs at p=400.”
- “The measured speedups hold across GPU architectures.”
- “The percolation threshold is universally 0.90 in cubic-work fraction.”

## 15. Suggested paper framing

### Possible title

**When Does GPU Batching Help Block-Sparse Graphical Lasso? A Reproducible
Study of Statistical Quality, Percolation, and Consumer-GPU Overhead**

### Central paper narrative

1. Connected-component decomposition can exactly reduce graphical-lasso work.
2. Profiling shows that sequential per-block orchestration and tiny linear
   algebra calls motivate batching.
3. A batched solver preserves the GGLasso solution numerically.
4. CPU batching is the strongest general result; GPU batching crosses over only
   at larger tested dimensions.
5. Statistical support quality and covariance percolation restrict when useful
   independent blocks are available.
6. Therefore the right research question is not merely “Can the eigensolver run
   on a GPU?” but “At a statistically defensible lambda, is there enough
   appropriately sized independent work to amortize GPU execution?”

### Recommended paper structure

1. Introduction and contributions
2. Graphical lasso and block decomposition
3. Profiling-driven batched ADMM design
4. Experimental methodology and reproducibility controls
5. Numerical equivalence
6. CPU/GPU performance and crossover
7. Percolation and statistical-quality analysis
8. Limitations and threats to validity
9. Related work
10. Conclusion

## 16. How to verify the project

Run from the repository root in Windows PowerShell.

### Fast correctness check

```powershell
.\.venv\Scripts\python.exe src\phase1_repro.py --out results\environment.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected result: six passing tests. The CUDA test may be skipped only on a
machine without an available CUDA device.

### Smoke benchmark

```powershell
.\.venv\Scripts\python.exe src\benchmark_phase3.py --quick `
  --out results\phase3_smoke.csv --machine primary --blas-threads 8
.\.venv\Scripts\python.exe src\analyze_phase3.py `
  results\phase3_smoke.csv --out-dir results\smoke
```

### Full reproduction

```powershell
.\run_final_experiments.ps1
```

The full run takes hours. Long stages use adjacent `.partial.jsonl`
checkpoints; rerunning the same command with `--resume` preserves completed
configurations.

## 17. Final artifact map

| Artifact | Meaning |
|---|---|
| `results/rq1_primary_1t.csv` | Corrected one-thread CPU sweep, 384 rows |
| `results/rq1_primary_8t.csv` | Corrected eight-thread CPU sweep, 384 rows |
| `results/percolation.csv` | Three-seed percolation study, 3,600 rows |
| `results/phase3_results.csv` | Full five-path comparison, 144 rows |
| `results/phase3_best_f1.csv` | Final 24 best-F1 operating points |
| `results/phase3_speedup.png` | Final grouped speedup figure |
| `src/batched_sgl.py` | Batched NumPy/PyTorch/CUDA implementation |
| `tests/test_batched_sgl.py` | Correctness and reproducibility tests |
| `notes/RUNBOOK.md` | Complete experiment protocol |
| `notes/findings.md` | Concise final findings and historical notes |

## 18. Bottom line

The project is complete through the requested implementation, experiment,
analysis, and documentation stages. The result is credible because it reports
both success and failure regions, selects performance points by statistical
quality, uses fair end-to-end timing, and establishes numerical equivalence.

Its strongest publishable message is not that GPUs always accelerate graphical
lasso. It is that **the usefulness of GPU batching is jointly determined by
problem dimension, detected component sizes, execution overhead, and the
regularization level required for statistical quality**. That is a more
interesting and more defensible research conclusion.
