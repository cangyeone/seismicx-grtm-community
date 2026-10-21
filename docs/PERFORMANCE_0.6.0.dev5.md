# Forward/derivative device and batch measurements — 0.6.0.dev5

English | [简体中文](PERFORMANCE_0.6.0.dev5.zh-CN.md)

2026-10-21; source/binary revision `44e0ef8`. Ratios use the same host only and mean baseline time / candidate time; below 1 means slower.
These results do not establish improved inverse/geological accuracy. See [device boundaries](DEVICE_EXECUTION.md) and [dev5 installation](PREVIEW_0.6.0.dev5.md).

Changes measured here: native CPU RF forward/analytic VJP, CUDA RF gradient-temporary reduction, CUDA Green analytic derivative integration/contraction and scratch reuse, ordered Green batch backward, and surface CPU model×period parallelism.
MPS RF/surface is an explicit CPU FP64 bridge; Green Metal remains hybrid. Interface compatibility is not counted as native GPU acceleration.


## Protocol and resources

Apple M4 Max, 16 cores, 128 GiB; remote Threadripper PRO 3955WX, 16 cores/32 threads, RTX 5090 (device 1, approximately 32 GiB), WSL2, driver 576.88.
Torch is 2.8.0 on Apple / 2.8.0+cu128 on CUDA. Three-family timing uses installed source-built wheels; compiled release wheels from the same revision are separately tested on real hardware.
pyfk uses its unmodified Cython build and matching GRTM native libraries/source wrapper, Apple Python 3.9.6 and remote 3.8.10; other measurements use 3.12/3.13.

Each operation has 1 warmup and 3 synchronized wall-time trials; tables use the median and raw repetitions are retained. Backend order is fixed, not randomized/rotated, so short calls remain sensitive to system noise.
Campaign jobs are serialized per host; unrelated user jobs are not stopped. No cross-host ratio is used.
Both RF CPU paths use 4 threads (Torch preparation also 4); surface CPU 16; Green total 16 host threads, split 4 model workers × 4 native threads for B>1.
Original disba/pyfk remain serial; no parallel baseline was invented. Multicore GRTM and original serial-library ratios are distinguished below.


## RF: 32 layers, 2048-point FFT

| B | Operation | Torch CPU ms | Native CPU ms | CUDA ms | Native CPU / CUDA | Torch CPU / CUDA |
|---|---|---|---|---|---|---|
| 1 | forward | 25.763 | 3.145 | 1.935 | 1.63× | 13.32× |
| 1 | forward vjp | 100.848 | 12.392 | 12.095 | 1.02× | 8.34× |
| 16 | forward | 92.198 | 36.689 | 1.975 | 18.58× | 46.69× |
| 16 | forward vjp | 279.688 | 113.743 | 12.234 | 9.30× | 22.86× |
| 64 | forward | 297.263 | 148.380 | 3.253 | 45.61× | 91.37× |
| 64 | forward vjp | 971.485 | 436.757 | 29.758 | 14.68× | 32.65× |
| 256 | forward | 1205.945 | 612.280 | 16.817 | 36.41× | 71.71× |
| 256 | forward vjp | 3504.537 | 1643.763 | 46.013 | 35.72× | 76.16× |


The improved native CPU baseline lowers the GPU ratio relative to the old Torch baseline; both are shown. CUDA RF propagation, preparation, FFT and derivatives stay on CUDA. VJP timing includes the forward and one scalar-loss gradient.

### Apple CPU only

| B | Operation | Torch ms | Native ms | CPU speedup |
|---|---|---|---|---|
| 1 | forward | 10.582 | 1.037 | 10.20× |
| 1 | forward vjp | 34.436 | 4.096 | 8.41× |
| 16 | forward | 68.904 | 13.335 | 5.17× |
| 16 | forward vjp | 179.101 | 45.355 | 3.95× |
| 64 | forward | 201.445 | 52.359 | 3.85× |
| 64 | forward vjp | 559.172 | 172.827 | 3.24× |
| 256 | forward | 751.843 | 203.138 | 3.70× |
| 256 | forward vjp | 2042.632 | 669.002 | 3.05× |


## Dispersion and kernels: 32 layers, 12 periods

| B | Operation | CPU16 ms | CUDA ms | CPU16 / CUDA | disba1 ms | disba1 / CUDA |
|---|---|---|---|---|---|---|
| 1 | phase forward | 3.790 | 11.758 | 0.32× | 1.497 | 0.13× |
| 1 | group forward | 3.884 | 13.337 | 0.29× | 2.949 | 0.22× |
| 1 | phase forward vs kernel | 8.985 | 18.955 | 0.47× | 227.881 | 12.02× |
| 1 | group forward vs kernel | 28.896 | 51.974 | 0.56× | 457.175 | 8.80× |
| 16 | phase forward | 4.535 | 12.620 | 0.36× | 23.690 | 1.88× |
| 16 | group forward | 5.704 | 14.606 | 0.39× | 47.123 | 3.23× |
| 16 | phase forward vs kernel | 56.155 | 22.769 | 2.47× | 3633.698 | 159.59× |
| 16 | group forward vs kernel | 327.254 | 66.229 | 4.94× | 7265.711 | 109.71× |
| 64 | phase forward | 16.245 | 19.575 | 0.83× | 94.797 | 4.84× |
| 64 | group forward | 17.715 | 20.921 | 0.85× | 189.128 | 9.04× |
| 64 | phase forward vs kernel | 191.955 | 43.252 | 4.44× | 14565.363 | 336.76× |
| 64 | group forward vs kernel | 1281.483 | 149.331 | 8.58× | 29143.324 | 195.16× |
| 256 | phase forward | 61.884 | 39.291 | 1.58× | 380.205 | 9.68× |
| 256 | group forward | 66.327 | 40.961 | 1.62× | 756.046 | 18.46× |
| 256 | phase forward vs kernel | 728.445 | 119.635 | 6.09× | 58549.993 | 489.41× |
| 256 | group forward vs kernel | 5041.847 | 473.128 | 10.66× | 117157.086 | 247.62× |


Rayleigh fundamental mode, periods 2–50 s. Kernel timing includes roots and all-layer Vs derivatives, not only a cotangent contraction. GRTM uses default root tolerance 1e-10 km/s and dual/hyper-dual plus implicit differentiation.
[disba 0.7.0](https://github.com/keurfonluu/disba) uses unchanged default Dunkin, dc=.005, group dt=.025 and kernel dp=.025, serial models. Its one-sided finite-difference kernel is not derivative ground truth; this external comparison is not tolerance matched.


### Apple CPU comparison

| B | Operation | disba1 ms | GRTM CPU16 ms | Ratio |
|---|---|---|---|---|
| 1 | phase forward | 0.833 | 0.932 | 0.89× |
| 1 | group forward | 1.727 | 1.013 | 1.71× |
| 1 | phase forward vs kernel | 135.496 | 2.512 | 53.93× |
| 1 | group forward vs kernel | 263.381 | 8.280 | 31.81× |
| 256 | phase forward | 212.123 | 23.627 | 8.98× |
| 256 | group forward | 424.558 | 26.856 | 15.81× |
| 256 | phase forward vs kernel | 34417.819 | 242.079 | 142.18× |
| 256 | group forward vs kernel | 69101.787 | 1571.252 | 43.98× |


## MPS compatibility: actual computation is CPU FP64

| Family | B | Operation | CPU ms | MPS bridge ms | Relative Δ |
|---|---|---|---|---|---|
| rf | 1 | forward | 1.037 | 4.346 | 4.358e-08 |
| rf | 1 | forward vjp | 4.096 | 8.381 | 1.215e-07 |
| rf | 256 | forward | 203.138 | 204.072 | 4.909e-08 |
| rf | 256 | forward vjp | 669.002 | 665.742 | 1.458e-07 |
| surface | 1 | phase forward | 0.932 | 1.928 | 1.980e-08 |
| surface | 1 | group forward | 1.013 | 1.847 | 2.826e-08 |
| surface | 1 | phase forward vs kernel | 2.512 | 4.829 | 2.938e-08 |
| surface | 1 | group forward vs kernel | 8.280 | 9.001 | 2.357e-08 |
| surface | 256 | phase forward | 23.627 | 23.798 | 2.194e-08 |
| surface | 256 | group forward | 26.856 | 25.347 | 2.226e-08 |
| surface | 256 | phase forward vs kernel | 242.079 | 256.411 | 2.544e-08 |
| surface | 256 | group forward vs kernel | 1571.252 | 1668.371 | 2.557e-08 |


These MPS rows include CPU double-precision solving and conversion of real outputs to MPS; they are not Apple GPU acceleration measurements. Float32 input/output rounding and call overhead remain in the records.

## Green forward and structural VJP

| Host | B | Operation | CPU16 ms | CUDA / Metal ms | Ratio |
|---|---|---|---|---|---|
| cuda | 1 | forward | 15.103 | 10.200 | 1.48× |
| cuda | 1 | forward vjp | 82.967 | 243.703 | 0.34× |
| cuda | 16 | forward | 101.733 | 77.862 | 1.31× |
| cuda | 16 | forward vjp | 1191.203 | 1116.048 | 1.07× |
| apple | 1 | forward | 4.276 | 4.987 | 0.86× |
| apple | 1 | forward vjp | 63.303 | 69.568 | 0.91× |
| apple | 16 | forward | 52.600 | 52.721 | 1.00× |
| apple | 16 | forward vjp | 961.419 | 956.680 | 1.00× |


8 layers, 4 receivers, 64 samples, frozen wavenumber grid, all-layer Vp/Vs parameters; VJP includes forward. CPU uses node-local reverse AD; CUDA integrates selected analytic directions and contracts on GPU, so cost grows with parameter count. Native CUDA does not guarantee speedup: CPU reverse AD can be preferable for small problems or many parameters. Slower cases are retained.
CUDA Green retains low-frequency long double and host FFT/physical synthesis; Metal structural derivatives remain CPU. Model batching is host scheduling, not a fused GPU model axis.


## Unmodified pyfk comparison: refined grids

| Host | L | B | pyfk1 ms | CPU1 ms | CPU16 ms | GPU ms | pyfk / CPU16 | pyfk / GPU | CPU16 / GPU | Relative Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| apple | 4 | 1 | 511.061 | 134.796 | 13.349 | 15.141 | 38.29× | 33.75× | 0.88× | 4.76e-05 |
| apple | 4 | 16 | 8240.526 | 2113.369 | 189.211 | 198.248 | 43.55× | 41.57× | 0.95× | 4.77e-05 |
| apple | 32 | 1 | 1024.982 | 858.673 | 90.262 | 92.607 | 11.36× | 11.07× | 0.97× | 4.66e-05 |
| apple | 32 | 16 | 16433.144 | 14051.394 | 1320.830 | 1370.228 | 12.44× | 11.99× | 0.96× | 4.67e-05 |
| cuda | 4 | 1 | 863.427 | 437.350 | 36.077 | 8.508 | 23.93× | 101.48× | 4.24× | 4.76e-05 |
| cuda | 4 | 16 | 13473.042 | 6674.520 | 544.670 | 79.323 | 24.74× | 169.85× | 6.87× | 4.77e-05 |
| cuda | 32 | 1 | 2468.222 | 1963.488 | 173.011 | 31.445 | 14.27× | 78.49× | 5.50× | 4.66e-05 |
| cuda | 32 | 16 | 39649.557 | 31444.450 | 2487.010 | 327.282 | 15.94× | 121.15× | 7.60× | 4.67e-05 |


The baseline is unchanged [pyfk](https://github.com/ziyixi/pyfk/tree/810e77dee6e7fe12ebac41832dfc17e6b7bab29f) `calculate_gf`. The table uses 4 receivers, 64 samples, spacing=.0125, Q=1e9 near-elastic models to avoid different finite-Q conventions, matching damping, taper, model, receiver geometry and conservative cutoff.
Time origins come directly from pyfk travel times; no amplitude, sign or delay fitting is used. The fixed force mapping follows source equations: GRTM first 5 channels = pyfk `[4,1,5,3,0]*[1,-1,-1,1,-1]`.
Original pyfk midpoint and GRTM trapezoid quadrature differ in nodes/rules, so this is not strictly equal-accuracy timing. Refinement below supplies convergence evidence. Coarse-grid throughput and 64-receiver cases are retained in raw JSON.
pyfk has no equivalent structural VJP in this interface; no external derivative speedup is invented.


| Host | L | Spacing | All5 relative Δ | Max channel relative Δ |
|---|---|---|---|---|
| apple | 4 | 0.0125 | 4.588e-05 | 8.719e-05 |
| apple | 4 | 0.00625 | 1.149e-05 | 2.183e-05 |
| apple | 4 | 0.003125 | 2.893e-06 | 5.489e-06 |
| apple | 32 | 0.0125 | 4.470e-05 | 8.410e-05 |
| apple | 32 | 0.00625 | 1.119e-05 | 2.105e-05 |
| apple | 32 | 0.003125 | 2.818e-06 | 5.292e-06 |
| cuda | 4 | 0.0125 | 4.588e-05 | 8.719e-05 |
| cuda | 4 | 0.00625 | 1.149e-05 | 2.183e-05 |
| cuda | 4 | 0.003125 | 2.893e-06 | 5.489e-06 |
| cuda | 32 | 0.0125 | 4.470e-05 | 8.410e-05 |
| cuda | 32 | 0.00625 | 1.119e-05 | 2.105e-05 |
| cuda | 32 | 0.003125 | 2.818e-06 | 5.292e-06 |


## Original pyfk CUDA comparison

| L | B | Receivers | pyfk CUDA ms | GRTM CUDA ms | pyfk / GRTM | pyfk CUDA / CPU relative Δ |
|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 7.996 | 3.785 | 2.11× | 2.811e-13 |
| 4 | 16 | 4 | 116.966 | 79.024 | 1.48× | 2.726e-13 |
| 4 | 64 | 4 | 469.202 | 79.838 | 5.88× | 2.680e-13 |
| 4 | 1 | 64 | 55.377 | 12.652 | 4.38× | 3.276e-13 |
| 32 | 1 | 4 | 14.654 | 25.685 | 0.57× | 4.170e-13 |
| 32 | 16 | 4 | 191.453 | 99.021 | 1.93× | 4.569e-13 |


This additional comparison uses the spacing=.1 throughput grid, separately from the refined table. Both use the same RTX5090. Original pyfk CUDA uses Numba/CuPy without mathematical changes; the last column checks its own CUDA/CPU agreement. Cross-solver discretization differences remain in the raw records. Some short GPU calls have substantial trial variation:4-layer/B16 GRTM CUDA ranges27.6–100.0ms. A median is not a guaranteed performance level.

## Numerical checks and limits

Largest relative norm discrepancy in this CUDA/CPU campaign:rf: 2.876e-15; surface: 2.740e-09; green: 2.907e-09.

| disba B256 | Relative Δ from GRTM |
|---|---|
| phase forward | 3.963e-07 |
| group forward | 6.598e-05 |
| phase forward vs kernel | 1.944e-02 |
| group forward vs kernel | 2.275e-02 |


These check this model cohort/discretization, not a universal error bound. Verification covers independent analytic/finite-difference checks, parameter JVP/VJP duality, nondefault CUDA devices/streams, MPS graph linkage and model mutation. The existing [strict surface audit](PERFORMANCE_0.6.0.dev4.md) remains available; this work does not change the physics or relax root tolerances.
MPS float32 input/real output adds rounding; the CPU FP64 bridge does not remove that boundary. Extreme quasistatic structural gradients, higher-order autograd and untested GPU architectures carry no additional guarantee.


Compiled-release full regression:Mac111 tests,94 passed/17 skipped;CUDA111 tests,100 passed/11 skipped. Skips reflect platforms or optional dependencies. Bilingual API examples ran successfully on CPU,MPS and CUDA.

The documented single-model Green derivative records 32 CUDA frequency bins,0 CPU bins and 8 analytic parameter directions. This example does not trigger the low-frequency extended-precision exception;host FFT/physical-synthesis boundaries remain. It does not imply all models avoid host computation.

## Data and reproduction

- [backends-apple.json](../verification/final-backends-20261021/backends-apple.json)

- [backends-cuda.json](../verification/final-backends-20261021/backends-cuda.json)

- [pyfk-apple.json](../verification/final-backends-20261021/pyfk-apple.json)

- [pyfk-cuda.json](../verification/final-backends-20261021/pyfk-cuda.json)

- [pyfk-refined-apple.json](../verification/final-backends-20261021/pyfk-refined-apple.json)

- [pyfk-refined-cuda.json](../verification/final-backends-20261021/pyfk-refined-cuda.json)

- [provenance.json](../verification/final-backends-20261021/provenance.json)


```sh
python benchmarks/forward-derivatives/benchmark_backends.py --device cuda:1 --disba --output backends-cuda.json
python benchmarks/forward-derivatives/benchmark_backends.py --device mps --disba --output backends-apple.json
python benchmarks/forward-derivatives/benchmark_pyfk.py --backend cuda --device 1 --profile refined --output pyfk-refined-cuda.json
python benchmarks/forward-derivatives/benchmark_pyfk.py --backend metal --profile refined --output pyfk-refined-apple.json
```


Install GRTM, Torch, disba and original pyfk dependencies first; the pyfk build environment must satisfy its upstream requirements. Output files must not exist, to avoid replacing evidence. Compare matching hosts, models, batches, algorithms and error settings. See [verification](../verification/performance-0.6.0.dev5/verification.json).
