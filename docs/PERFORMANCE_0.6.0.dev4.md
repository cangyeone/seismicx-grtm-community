# Green functions, RF and dispersion: CPU/CUDA performance — 0.6.0.dev4

English | [简体中文](PERFORMANCE_0.6.0.dev4.zh-CN.md)

New: [2026-10-21 batch scaling](BATCH_SCALING_20261021.md) and
[model-batch API guide](BATCH_EXECUTION.md). The dated measurements below are
retained; the new sweep uses different sampled hardware load and is not mixed
into these timing ratios.

The optimized GRTM search preserves the requested precision. Batched dispersion
is substantially faster than the previous GRTM implementation; CPU threading
and CUDA have different break-even points. **Green structural differentiation
is still CPU work, including pipelines with CUDA forward.** RF and surface-wave
CUDA derivatives are native GPU calculations. No measured speedup is a claim
about posterior convergence or geological-model accuracy.

## Matched GRTM forward/derivative measurements

2026-10-20; Threadripper PRO 3955WX (16 physical / 32 logical CPUs), RTX 5090
32 GB (`cuda:1`), Linux/WSL, Python 3.13.5, Torch 2.8.0+cu128. The CPU and GPU
were shared with other workloads; no unrelated jobs were stopped. FP64 and
complex128 throughout. Prepared operators, one warmup, five whole-call host
wall-time trials and their median; CUDA synchronization surrounds each trial.
Construction/model upload are excluded from this table. Public validation,
host orchestration and output handling are included. The JSON retains every
trial and separately measured setup. Operation order rotates between trials;
backends run sequentially. Absolute times and speedups depend on current load.

Green: displacement in the time domain, one Earth model, 4 receivers, 64 samples, fixed 20205-node quadrature,
Vp/Vs in all 32 layers. Directional JVP and scalar adjoint VJP avoid a dense
Jacobian. RF: 16 models, 32 layers, one ray, 6 Gaussian bands, 81 samples,
nfft=2048, 41 polarization windows; VJP of fixed RF/apparent-Vs cotangents
w.r.t. top/Vp/Vs/density. The first top is fixed. Dispersion: 16 models,
Rayleigh fundamental, 12 periods from 2–50 s; the Vs kernel contains all layers.
These are different derivative tasks; their absolute times are not interchangeable.

| Task (32 layers) | CPU 1 (s) | CPU multi (s) | CUDA / hybrid † (s) | CPU 1 / target | CPU multi / target |
| --- | --- | --- | --- | --- | --- |
| Green / B=1 forward | 0.738855 | 0.178445 | 0.342739 | 2.16× | 0.52× |
| Green / B=1 JVP only † | 1.654211 | 0.407322 | 0.429144 | 3.85× | 0.95× |
| Green / B=1 VJP only † | 7.721161 | 1.430158 | 1.281868 | 6.02× | 1.12× |
| Green / B=1 forward + VJP † | 8.528672 | 1.403220 | 1.754349 | 4.86× | 0.80× |
| RF / B=16 forward | 0.442220 | 0.421455 | 0.024811 | 17.82× | 16.99× |
| RF / B=16 VJP only | 0.674402 | 0.741328 | 0.023473 | 28.73× | 31.58× |
| RF / B=16 forward + VJP | 1.085292 | 1.103559 | 0.052522 | 20.66× | 21.01× |
| Phase / B=16 forward | 0.056294 | 0.014906 | 0.039368 | 1.43× | 0.38× |
| Phase / B=16 forward + full Vs kernel | 0.549999 | 0.120235 | 0.060831 | 9.04× | 1.98× |
| Group / B=16 forward | 0.063379 | 0.019523 | 0.039908 | 1.59× | 0.49× |
| Group / B=16 forward + full Vs kernel | 3.902394 | 0.557266 | 0.148562 | 26.27× | 3.75× |

Ratios below 1 mean slower. † The Green target uses CUDA forward with 16 CPU
threads; its JVP/VJP runs on CPU. These entries do not measure CUDA structural
AD. Other CUDA rows use one host Torch thread and native GPU propagation and
derivatives. CPU multi means 16 workers for Green/dispersion, 4 Torch threads
for RF, not a single-core algorithm speedup. An initial RF16 run was stopped
after several minutes in the 8-layer/B16 case; its 12 completed preliminary
records are retained separately, with no fabricated timeout per-call timings.
`vjp_only` reuses the base linearization/graph; `forward + VJP` rebuilds
it. They are measured separately, so changing shared load can make individual
medians non-additive. No timings are obtained by subtracting noisy totals.

All 8-layer and single-model cases, both Green CPU/CUDA host-thread settings,
and all individual trials are in the comprehensive JSON. Small GPU tasks can
be slower than CPU; the 32-layer batched rows are not universal speed guarantees.

## Dispersion alone versus unchanged disba

32 layers, 12 periods; separate warmed five-trial campaign. Each CUDA ratio uses
its own interleaved disba timing from that campaign. disba 0.7.0 remains
single-threaded and unmodified: Dunkin, dc=.005, group dt=.025. GRTM has root
tolerance 1e-10 km/s and analytic group derivatives. **These are the original
default methods, not equal-error-budget timings.** disba's native root stop uses
relative 1e-6 speed; dc is the search spacing, not its final root error.

Single-thread CPU:

| B | kind | disba CPU1 (s) | GRTM CPU1 (s) | speedup |
| --- | --- | --- | --- | --- |
| 1 | phase | 0.002621 | 0.005037 | 0.52× |
| 1 | group | 0.005139 | 0.003489 | 1.47× |
| 16 | phase | 0.044729 | 0.059479 | 0.75× |
| 16 | group | 0.082886 | 0.061486 | 1.35× |
| 64 | phase | 0.167922 | 0.238703 | 0.70× |
| 64 | group | 0.342791 | 0.250233 | 1.37× |

CPU model workers = 16; baseline remains original serial disba:

| B | kind | disba CPU1 (s) | GRTM CPU16 (s) | speedup |
| --- | --- | --- | --- | --- |
| 16 | phase | 0.043589 | 0.021545 | 2.02× |
| 16 | group | 0.086080 | 0.019837 | 4.34× |
| 64 | phase | 0.168736 | 0.046853 | 3.60× |
| 64 | group | 0.341671 | 0.053518 | 6.38× |

CUDA and its own interleaved serial disba baseline:

| B | kind | disba prepared (s) | CUDA prepared (s) | speedup | disba full (s) | CUDA full (s) | full speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | phase | 0.002694 | 0.038230 | 0.07× | 0.002739 | 0.060934 | 0.04× |
| 1 | group | 0.005248 | 0.040977 | 0.13× | 0.005633 | 0.062361 | 0.09× |
| 16 | phase | 0.043558 | 0.040237 | 1.08× | 0.046289 | 0.057985 | 0.80× |
| 16 | group | 0.084485 | 0.042696 | 1.98× | 0.083519 | 0.065132 | 1.28× |
| 64 | phase | 0.172809 | 0.054301 | 3.18× | 0.171897 | 0.054602 | 3.15× |
| 64 | group | 0.346198 | 0.055410 | 6.25× | 0.338251 | 0.078363 | 4.32× |

CUDA full calls start from a host velocity model and include construction,
transfer and mesh planning. Prepared calls reuse resident models and meshes.
GRTM CPU16 is explicit model-level parallelism against unchanged serial disba.
For 32 layers/B16, the previous GRTM CPU phase/group medians were
3.303400/3.459268 s; the new medians are 0.059479/0.061486 s (about 55.5×/56.3×).
Previous CUDA medians were .225457/.258540 s, now .040237/.042696 s
(about 5.6×/6.1×). Old/new results are separate campaigns on the same shared
hardware, not paired trials. CPU phase at one thread still trails default disba
for this workload. B1 CUDA is dominated by latency; B64 amortizes it.

## Complete Vs kernels versus unchanged disba

32 layers/B16/12 periods, including root search and every layer's Vs derivative;
five warmed repetitions. CPU and CUDA comparison campaigns have separate disba
columns, so their baseline medians are not mixed.

| kind | disba CPU1 (s) | GRTM CPU1 (s) | CPU speedup | disba CPU1, CUDA campaign (s) | CUDA (s) | CUDA speedup |
| --- | --- | --- | --- | --- | --- | --- |
| phase | 6.944697 | 0.571227 | 12.16× | 6.850183 | 0.052390 | 130.75× |
| group | 13.634333 | 3.909884 | 3.49× | 13.643346 | 0.134783 | 101.22× |

disba uses its unchanged 2.5% one-sided model perturbation for these kernels;
GRTM uses analytic dual/hyper-dual derivatives. Kernel differences from disba
are not independent accuracy bounds. The full-kernel timings must not be quoted
as pure-dispersion speedups.

## Accuracy and execution checks

129 models, 4339 available modal points: normal gradients, LVZs, thin layers,
density contrasts and thick layers/short periods; both wave types, modes 0/1,
missing-mode masks, unordered/repeated periods and group velocities. Optimized
phase roots differ from the original complex full scan at stricter 1e-12 km/s
tolerance by at most 1.73e-12 km/s. The independent 70-digit audit uses general
4×4 eigenmatrix inversions and six exterior minors, not the fast five-state
recurrence. Its 48 phase/group cases give:

| Method | Cases | Max phase error (km/s) | Max group error (km/s) |
| --- | --- | --- | --- |
| GRTM auto CUDA | 48 | 1.502e-12 | 1.202e-10 |
| Original complex scan CPU | 48 | 1.783e-11 | 1.358e-09 |
| Original disba CPU (available cases) | 45 | 2.312e-06 | 1.438e-03 |

The original complex scan in this 70-digit sub-audit uses its default 1e-10
root tolerance. disba is available on 45 common cases: one two-layer Love curve
at 3 periods raises DispersionError in both phase and group calls. These failures
are retained, never converted to zero error or silently removed. Across all 48
GRTM cases and the 45 common cases the stated maxima are the same. This supports
precision on the tested models, not universal modal completeness, a rigorous
floating-point error bound, or greater geological inversion accuracy.

The comprehensive installed-wheel benchmark also records CPU/CUDA parity,
adjoint identities and central-difference checks (differences are validation only):

| Family | Metric | Maximum over benchmark cases |
| --- | --- | --- |
| green | adjoint_identity_relative_error | 6.925e-12 |
| green | forward_relative_cpu_error | 3.753e-13 |
| green | jvp_central_difference_relative_error | 4.572e-07 |
| green | vjp_relative_cpu_error | 1.724e-19 |
| receiver_function | directional_vjp_central_difference_relative_error | 1.687e-08 |
| receiver_function | rf_max_abs_cpu_error | 5.204e-16 |
| receiver_function | rf_relative_cpu_error | 1.159e-15 |
| receiver_function | vjp_relative_cpu_error | 2.575e-15 |
| dispersion | forward_relative_cpu_error | 2.118e-11 |
| dispersion | forward_vs_kernel_relative_cpu_error | 2.371e-08 |

The 10 binary build jobs passed installed CPU suites (105 discovered tests each,
63 Linux / 64 macOS passes; 42/41 optional/platform skips). On actual devices,
Linux/CUDA passed 96/105 with 9 skips; macOS/Metal passed 89/105 with 16 skips.
See [binary verification](../verification/preview-0.6.0.dev4.json)
for scope and the documented test-only Metal concurrency roundoff adjustment.

## Reproduce and inspect

Install the [matching dev4 binary](PREVIEW_0.6.0.dev4.md); stable PyPI is still
0.5.1. The binary source commit is `fbf130f9ce259faa89df6557c002dfb16acc8230`.
The comprehensive results use the installed CI wheel, not a source import.

```sh
python benchmarks/forward-derivatives/benchmark_forward_derivatives.py --device cuda:1 --threads 1 16 --layers 8 32 --batches 1 16 --repeat 5 --output comprehensive.json
python benchmarks/surface-wave/benchmark_surface_cpu.py --threads 1 --layers 32 --batches 16 --tasks sensitivity --repeat 5 --output kernels-cpu.json
python benchmarks/surface-wave/benchmark_surface_wave.py --device cuda:1 --layers 32 --batches 16 --repeat 5 --output kernels-cuda.json
python benchmarks/surface-wave/benchmark_surface_dispersion.py --device cuda:1 --repeat 5 --output dispersion-cuda.json
```

Install benchmark-only `disba==0.7.0` for the three external comparisons. GRTM
production does not depend on disba. The private implementation repository also
contains `examples/validate_surface_search.py --models 128 --mp-models 8` for the
70-digit audit and [surface-wave API/math notes](SURFACE_WAVES.md).

- [comprehensive-dev4.json](../verification/performance-0.6.0.dev4/comprehensive-dev4.json)
- [comprehensive-preliminary-rf16.json](../verification/performance-0.6.0.dev4/comprehensive-preliminary-rf16.json)
- [comprehensive-rf-surface-dev4.json](../verification/performance-0.6.0.dev4/comprehensive-rf-surface-dev4.json)
- [search-cpu-final.json](../verification/performance-0.6.0.dev4/search-cpu-final.json)
- [search-cpu16-final.json](../verification/performance-0.6.0.dev4/search-cpu16-final.json)
- [search-cuda-final.json](../verification/performance-0.6.0.dev4/search-cuda-final.json)
- [cpu-validation.json](../verification/performance-0.6.0.dev4/cpu-validation.json)
- [search-final-validation.json](../verification/performance-0.6.0.dev4/search-final-validation.json)
- [kernels-cpu-disba.json](../verification/performance-0.6.0.dev4/kernels-cpu-disba.json)
- [kernels-cuda-disba.json](../verification/performance-0.6.0.dev4/kernels-cuda-disba.json)
