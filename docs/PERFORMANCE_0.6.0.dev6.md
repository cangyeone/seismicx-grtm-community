# Phase and group velocity performance: 0.6.0.dev6

Same Threadripper PRO 3955WX (16 cores) and RTX 5090, CUDA device 1, FP64. 8/32 layers, 12 periods (2–50 s), fundamental Rayleigh mode. CPU budgets are 1 and 16 threads; CUDA roots and derivatives stay on GPU. Warmed, synchronized whole-batch wall time, median of 9 repetitions. Prepared models/grids; roots are recalculated each call with no solution cache. Full construction calls are reported separately. Baseline is the released dev5 binary; candidate is a local Release build of source commit `54cf35f`.

## CUDA before and after

| L | B | Kind | dev5 ms | dev6 ms | dev5/dev6 | CPU1/dev6 | CPU16/dev6 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 1 | phase | 1.538 | 1.332 | 1.15× | 0.45× | 0.31× |
| 8 | 1 | group | 1.968 | 1.620 | 1.22× | 0.27× | 0.27× |
| 8 | 32 | phase | 3.789 | 1.972 | 1.92× | 7.55× | 1.30× |
| 8 | 32 | group | 4.215 | 2.260 | 1.86× | 6.65× | 1.14× |
| 8 | 256 | phase | 8.740 | 4.836 | 1.81× | 24.84× | 2.67× |
| 8 | 256 | group | 9.180 | 5.101 | 1.80× | 24.18× | 2.77× |
| 32 | 1 | phase | 11.398 | 3.775 | 3.02× | 0.75× | 0.60× |
| 32 | 1 | group | 13.290 | 4.981 | 2.67× | 0.48× | 0.48× |
| 32 | 32 | phase | 14.730 | 6.744 | 2.18× | 11.61× | 1.13× |
| 32 | 32 | group | 16.581 | 7.932 | 2.09× | 10.09× | 1.10× |
| 32 | 256 | phase | 39.107 | 15.724 | 2.49× | 40.97× | 3.69× |
| 32 | 256 | group | 40.910 | 16.834 | 2.43× | 38.95× | 3.55× |

Single-model GPU calls can still be slower than CPU. Batch speedup depends on the CPU thread budget; a serial-CPU ratio is not a 16-thread-CPU ratio.

## CPU only (16-thread budget)

| L | B | Kind | dev5 ms | dev6 ms | dev5/dev6 |
|---:|---:|---|---:|---:|---:|
| 8 | 1 | phase | 0.477 | 0.412 | 1.16× |
| 8 | 1 | group | 1.293 | 0.432 | 2.99× |
| 8 | 32 | phase | 2.465 | 2.559 | 0.96× |
| 8 | 32 | group | 3.737 | 2.582 | 1.45× |
| 8 | 256 | phase | 14.131 | 12.914 | 1.09× |
| 8 | 256 | group | 15.766 | 14.154 | 1.11× |
| 32 | 1 | phase | 3.371 | 2.278 | 1.48× |
| 32 | 1 | group | 4.209 | 2.398 | 1.76× |
| 32 | 32 | phase | 8.029 | 7.643 | 1.05× |
| 32 | 32 | group | 9.673 | 8.745 | 1.11× |
| 32 | 256 | phase | 62.914 | 58.075 | 1.08× |
| 32 | 256 | group | 65.325 | 59.705 | 1.09× |

## disba 0.7.0 forwards and full calls

| L | B | Kind | disba1 prepared ms | CUDA prepared ms | disba1/CUDA | disba1 full ms | CUDA full ms | Full ratio |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | 1 | phase | 0.475 | 1.390 | 0.34× | 0.498 | 1.856 | 0.27× |
| 8 | 1 | group | 0.922 | 1.650 | 0.56× | 0.932 | 2.219 | 0.42× |
| 8 | 32 | phase | 14.350 | 2.028 | 7.08× | 14.457 | 2.884 | 5.01× |
| 8 | 32 | group | 28.420 | 2.316 | 12.27× | 28.474 | 3.165 | 9.00× |
| 8 | 256 | phase | 113.437 | 5.302 | 21.40× | 114.436 | 10.289 | 11.12× |
| 8 | 256 | group | 226.312 | 5.731 | 39.49× | 226.736 | 10.916 | 20.77× |
| 32 | 1 | phase | 1.483 | 4.015 | 0.37× | 1.481 | 4.654 | 0.32× |
| 32 | 1 | group | 2.940 | 5.344 | 0.55× | 2.936 | 5.884 | 0.50× |
| 32 | 32 | phase | 47.416 | 6.918 | 6.85× | 47.590 | 8.107 | 5.87× |
| 32 | 32 | group | 93.729 | 8.430 | 11.12× | 93.911 | 9.750 | 9.63× |
| 32 | 256 | phase | 375.876 | 16.341 | 23.00× | 376.046 | 25.636 | 14.67× |
| 32 | 256 | group | 748.594 | 17.842 | 41.96× | 751.288 | 27.214 | 27.61× |

Full calls include operator construction, host-model transfer and grid planning. Unchanged single-thread disba uses Dunkin, dc=.005 and group dt=.025; GRTM group velocity uses analytic period derivatives. Separate timing-run variation is retained in raw JSON. Maximum phase discrepancy is 2.45e-6 km/s and group discrepancy 4.55e-4 km/s. The latter includes disba period-difference error and is not an accuracy verdict on GRTM. disba higher-mode sensitivities are not treated as ground truth.

## Accuracy and implementation

33 models and 1,130 valid mode/period points include LVZs, density contrasts, thin layers, thick layers at short periods, shuffled and duplicate periods. CPU and CUDA pass original complex-GRTM full-scan comparisons; maximum phase discrepancy is 1.73e-12 km/s. Eight-model independent 70-digit matrix reference: phase error ≤1.51e-12 km/s, group error ≤1.21e-10 km/s. New mode-1/2 Vs-kernel finite differences (including Love near cutoff at 8 s) pass rtol=atol=1e-6. All 17 surface tests pass on CUDA; Apple CPU passes 14 with 3 CUDA skips.

Changes: tighter real trial roots avoid complex Newton repairs; CUDA separates scan/trial/complex bracket certification, with full on-device fallback for ambiguous cases. Large fundamental-mode batches stop ascending scans early; small batches retain full-grid parallelism. Thin-layer polynomials reduce to four state-map applications at unchanged Taylor order. Small CPU group passes avoid excess worker launches. FP64, mode indexing, search mesh, 1e-8 complex residual gate and user root tolerance are unchanged.

For both velocities, call `result = sw.dispersion(periods, kind="group")` once and read `result["phase_velocity"]` and `result["velocity"]`; the root search runs once.

This change concerns surface waves. Green CUDA reverse AD and FFT/physical synthesis optimization are separate work and are not included in these ratios.

Raw evidence and reproduction:

- [before-cuda.json](../verification/performance-0.6.0.dev6/surface/before-cuda.json)
- [after-cuda.json](../verification/performance-0.6.0.dev6/surface/after-cuda.json)
- [disba-cuda.json](../verification/performance-0.6.0.dev6/surface/disba-cuda.json)
- [validation-apple.json](../verification/performance-0.6.0.dev6/surface/validation-apple.json)
- [validation-cuda.json](../verification/performance-0.6.0.dev6/surface/validation-cuda.json)
