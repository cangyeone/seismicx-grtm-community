# Surface-wave dispersion and sensitivity timing — 0.6.0.dev3

2026-10-20; RTX 5090, Threadripper PRO 3955WX; FP64/complex128.
Main tables use **32 layers, 12 periods (2–50 s), 16 models, fundamental Rayleigh**.
Times are seconds for a complete batch.

## Pure CPU versus original disba 0.7.0

Both are single-threaded. GRTM uses semi-analytic derivatives. Disba keeps its
original default Dunkin implementation, without added threading. Three warmed
repetitions, medians. Speedup is disba time / GRTM time; below 1 means GRTM is slower.

| Task | Kind | disba CPU (s) | GRTM CPU (s) | Speedup |
|---|---|---:|---:|---:|
| dispersion | phase | 0.042797 | 3.303400 | 0.0130× |
| dispersion | group | 0.084155 | 3.459268 | 0.0243× |
| sensitivity | phase | 7.486630 | 4.128910 | 1.8132× |
| sensitivity | group | 14.863389 | 9.367613 | 1.5867× |

Complete Vs kernels are **1.81× / 1.59×** faster on CPU for phase/group. Dispersion
alone has no speedup: GRTM CPU is about 77× / 41× slower. GRTM currently scans a
complete adaptive speed grid per period; disba uses prior period/mode results to
narrow its root search. These are library-default comparisons, not matched root
tolerances: GRTM uses 1e-10 km/s, disba dc=.005. Disba group speed uses its default
frequency difference dt=.025; GRTM uses an analytic period derivative.
Kernel speedups must not be used as dispersion-only speedups.

## CUDA dispersion only, without kernels

Five warmed repetitions, synchronized host wall time. Prepared calls include
all root calculations. Full calls also include construction from a host velocity
model, model transfer and search-grid planning.

| Kind | disba CPU (s) | CUDA prepared (s) | CPU/CUDA | CUDA full call (s) | Full-call CPU/CUDA |
|---|---:|---:|---:|---:|---:|
| phase | 0.043280 | 0.225457 | 0.1920× | 0.257796 | 0.1642× |
| group | 0.086314 | 0.258540 | 0.3339× | 0.303097 | 0.2758× |

CUDA dispersion is currently slower than disba CPU. Comparing identical GRTM
mathematics instead gives the following CPU/CUDA speedups for prepared calls:

| Kind | GRTM CPU (s) | GRTM CUDA (s) | Speedup |
|---|---:|---:|---:|
| phase | 3.133448 | 0.222995 | 14.05× |
| group | 3.373608 | 0.256063 | 13.17× |

## Relation to complete sensitivity-kernel timings

A separate final CMake-library campaign measured **27.4× / 41.1×** CUDA speedups
over disba CPU for complete phase/group Vs kernels, and **16.0× / 27.6×** against
identical GRTM CPU mathematics. Raw repetitions are retained. Those timings and
the pure CPU supplement above are separate campaigns; their times are not mixed
to construct new ratios.

The host and GPUs were shared; no other jobs were stopped. Results describe
these models, defaults and hardware load, without inferring geological inversion
accuracy. Phase speeds differed from disba by at most about 2.44e-6 km/s in these
benchmarks; group differences include disba's default frequency approximation.

[Raw verification data](../verification/surface-wave-0.6.0.dev3/) includes single
models, batches and individual repetitions. See the [API and method guide](SURFACE_WAVES.md).
