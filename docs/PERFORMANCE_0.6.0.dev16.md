# Noise-correlation CPU/CUDA performance (0.6.0.dev16)

[中文](PERFORMANCE_0.6.0.dev16.zh-CN.md) · [API](NOISE_CORRELATION.md) · [Raw timing samples](../verification/preview-0.6.0.dev16/performance.json) · [Public-API benchmark](../benchmarks/data-processing/benchmark_noise_correlation.py)

Measured 2026-11-10 using the installed Linux CPython 3.13 compiled wheel on
AMD Ryzen Threadripper PRO 3955WX (16 cores / 32 logical CPUs) and RTX 5090,
PyTorch 2.8.0+cu128 / CUDA 12.8 / NumPy 2.1.3. The server is shared; existing
jobs were left running. Each CPU/GPU comparison uses the same data and settings.
CPU is the fastest measured alternative among NumPy, Torch 1 thread and Torch
8 threads, not a claim of the fastest possible CPU implementation or thread count.

Whole pipeline: mean removal, 5% cosine taper at each end, smooth whitening
(11 bins, 0.2–10 Hz, 0.1 Hz shoulders), coefficient normalization, all i<j
linear CCFs, weighted mean stack. Sampling interval 0.02 s, 50% window overlap,
±5 s output, 8-window/256-pair blocks, prevalidated input (`check_finite=False`).
This table measures linear stacking; it does not measure PWS or backward time.
Warm-up is excluded (2 CPU calls; CUDA at least 8 calls and 0.25 s). Values are
medians of 3 synchronized wall-clock runs, including Python dispatch and device
allocations; all individual samples are retained. No disk I/O is included.

| Case | Dtype | Fastest tested CPU ms | Resident CUDA ms | CUDA + transfers ms | Resident speedup | With transfers |
|---|---|---:|---:|---:|---:|---:|
| small | float32 | 1.000 | 2.642 | 2.699 | 0.38× | 0.37× |
| small | float64 | 1.084 | 2.644 | 2.738 | 0.41× | 0.40× |
| array | float32 | 265.478 | 9.893 | 10.956 | 26.83× | 24.23× |
| array | float64 | 418.939 | 12.278 | 14.191 | 34.12× | 29.52× |
| batch4 | float32 | 916.556 | 18.154 | 21.594 | 50.49× | 42.44× |
| batch4 | float64 | 1737.672 | 37.670 | 43.268 | 46.13× | 40.16× |
| dense128 | float32 | 1935.915 | 33.407 | 34.901 | 57.95× | 55.47× |
| dense128 | float64 | 3556.845 | 67.575 | 72.328 | 52.64× | 49.18× |

| Case | B | Stations | Record samples | Window samples | Windows | Pairs per batch |
|---|---:|---:|---:|---:|---:|---:|
| small | 1 | 4 | 2048 | 512 | 7 | 6 |
| array | 1 | 32 | 65536 | 4096 | 31 | 496 |
| batch4 | 4 | 32 | 65536 | 4096 | 31 | 496 |
| dense128 | 1 | 128 | 32768 | 4096 | 15 | 8128 |


CUDA + transfers includes waveform upload and CCF download; pair/lag metadata stays
on GPU. Small tasks lose to CPU because launch/dispatch costs dominate. Larger
arrays/batches benefit from FFT reuse, pairwise GPU products, and frequency-domain
stacking before the inverse FFT. These are workload-specific speedups over this
implementation's measured CPU paths; no FastXC/NoisePy performance claim is made.

Maximum CUDA-vs-NumPy relative L2 error: **FP32 4.13e-07**, **FP64 1.24e-15**.
These compare coefficient-normalized CCFs on the listed finite cases, not a
universal error guarantee. Peak allocated CUDA memory ranges from about 1 MiB
(small FP32) to 2.84 GiB (B=4 FP64); detailed values are in the JSON.
Linear stacks retain frequency accumulators for all output pairs as well as
block temporaries; specifying fewer pairs is necessary for very large arrays.

Installed-wheel regression: macOS 190 passed / 46 skipped; Linux/CUDA 214 passed /
22 skipped, 236 tests each. All 15 correlation tests pass on CUDA (Mac skips the
CUDA-only test). See [verification](../verification/preview-0.6.0.dev16/verification.json).
The GitHub Actions run was cancelled; these results are from actual installed
wheels on the two machines.

For comparisons to `source_field.receiver_cross_spectrum`, note its convention
is `E[Y_i conj(Y_j)]`; this CCF uses `conj(Y_i)*Y_j`. Transpose/conjugate the
expected spectral indices and account for one-sided PSD/FFT scaling. Nonlinear
whitening and finite-window processing also change the statistical target.
