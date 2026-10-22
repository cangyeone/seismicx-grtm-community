# FP64 eigenfunction and forced adjoints: performance and validation — dev8

[Methods/API](VARIATIONAL_KERNELS.md) · [Installation / ABI limits](PREVIEW_0.6.0.dev8.md)

Actual installed compiled wheels; four native CPU threads and one Torch thread, RTX 5090 GPU 1 with explicit synchronization. FP64, two warmups, median of three measurements. Surface workloads include eight periods and all thickness/Vp/Vs/density kernels, native units, fundamental Rayleigh mode, root tolerance 1e-11 km/s. A 32-layer model has 128 entries, including zero half-space thickness. Raw data retain every case and timing sample.

## Surface waves: methods within the same dev8 binary

Full-call ratios include roots; kernel ratios exclude roots but include the root-path base derivative. Ratios are existing characteristic-adjoint time divided by new energy time; below one means a slowdown. The default remains `adjoint`; select `differentiation="eigenfunction"` explicitly.

### Apple M4 Max CPU

| Layers | Batch | Kind | Adjoint ms | Energy ms | Full ratio | Kernel ratio |
|---|---|---|---:|---:|---:|---:|
| 8 | 1 | phase | 0.436 | 0.375 | 1.16× | 2.08× |
| 8 | 1 | group | 1.837 | 0.322 | 5.70× | 18.49× |
| 8 | 32 | phase | 6.056 | 2.812 | 2.15× | 4.25× |
| 8 | 32 | group | 46.080 | 3.251 | 14.18× | 31.83× |
| 32 | 1 | phase | 1.605 | 1.150 | 1.40× | 3.32× |
| 32 | 1 | group | 21.061 | 1.197 | 17.59× | 94.23× |
| 32 | 32 | phase | 25.854 | 11.935 | 2.17× | 4.47× |
| 32 | 32 | group | 647.771 | 13.336 | 48.57× | 113.31× |
| 96 | 1 | phase | 5.121 | 3.774 | 1.36× | 4.11× |
| 96 | 1 | group | 180.965 | 3.938 | 45.96× | 292.62× |

### Threadripper PRO 3955WX CPU

| Layers | Batch | Kind | Adjoint ms | Energy ms | Full ratio | Kernel ratio |
|---|---|---|---:|---:|---:|---:|
| 8 | 1 | phase | 1.649 | 1.177 | 1.40× | 1.92× |
| 8 | 1 | group | 8.466 | 1.962 | 4.32× | 18.17× |
| 8 | 32 | phase | 19.580 | 9.767 | 2.00× | 4.32× |
| 8 | 32 | group | 219.068 | 14.891 | 14.71× | 39.21× |
| 32 | 1 | phase | 6.336 | 5.350 | 1.18× | 3.05× |
| 32 | 1 | group | 98.833 | 3.789 | 26.08× | 109.16× |
| 32 | 32 | phase | 85.986 | 43.114 | 1.99× | 4.70× |
| 32 | 32 | group | 3062.461 | 52.729 | 58.08× | 148.29× |
| 96 | 1 | phase | 15.240 | 10.359 | 1.47× | 4.44× |
| 96 | 1 | group | 852.870 | 11.504 | 74.13× | 340.09× |

### RTX 5090 CUDA

| Layers | Batch | Kind | Adjoint ms | Energy ms | Full ratio | Kernel ratio |
|---|---|---|---:|---:|---:|---:|
| 8 | 1 | phase | 3.692 | 3.776 | 0.98× | 0.99× |
| 8 | 1 | group | 13.458 | 4.848 | 2.78× | 3.44× |
| 8 | 32 | phase | 3.956 | 4.067 | 0.97× | 0.99× |
| 8 | 32 | group | 13.695 | 5.171 | 2.65× | 3.42× |
| 32 | 1 | phase | 12.203 | 13.396 | 0.91× | 0.88× |
| 32 | 1 | group | 54.148 | 18.992 | 2.85× | 3.32× |
| 32 | 32 | phase | 13.868 | 14.758 | 0.94× | 0.91× |
| 32 | 32 | group | 56.187 | 20.314 | 2.77× | 3.37× |
| 96 | 1 | phase | 34.713 | 39.515 | 0.88× | 0.84× |
| 96 | 1 | group | 160.865 | 56.931 | 2.83× | 3.23× |

Same-host CPU/CUDA ratios for the new energy method, 32 layers and B=32:

- phase: **2.92×**.

- group: **2.60×**.

Large CPU group ratios are associated with the old parameter-wise analytic guard near critical modal coordinates. Measured roots and the old threshold identify 1 of 8 periods for both 8/32 layers and 3 of 8 periods for 96 layers. The energy method needed no fallback in these cases; these ratios are not universal. B=32 CUDA phase kernels are still slightly slower end to end, while group improves. Root search and launch costs matter. Apple-to-remote-GPU ratios are not same-host speedups.

## RF: dev7 versus dev8 forward plus adjoint

Includes a joint RF/polarization objective, not just the material contraction. Release build environments differ, so small changes cannot be attributed entirely to the new contraction.

| Host | Layers | Batch | dev7 ms | dev8 ms | Ratio |
|---|---|---|---:|---:|---:|
| apple | 8 | 1 | 0.738 | 0.780 | 0.95× |
| apple | 8 | 32 | 7.316 | 6.965 | 1.05× |
| apple | 32 | 1 | 1.381 | 1.207 | 1.14× |
| apple | 32 | 32 | 26.276 | 22.875 | 1.15× |
| cuda | 8 | 1 | 5.380 | 5.144 | 1.05× |
| cuda | 8 | 32 | 5.249 | 5.230 | 1.00× |
| cuda | 32 | 1 | 5.979 | 5.900 | 1.01× |
| cuda | 32 | 32 | 5.940 | 5.906 | 1.01× |

## Green functions: many parameters versus one

32 layers, B=4, 16 samples and 32 fixed wavenumber steps. This compares the retained analytic adjoint against per-parameter analytic tangents, not a new dev8 Green volume-integral or reverse engine. CUDA tangents include legacy host synthesis; adjoints use the existing GPU FFT/synthesis, so this is a pipeline comparison. One-parameter adjoints can be slower.

| Host | Parameters | Tangent ms | Adjoint ms | Ratio |
|---|---|---:|---:|---:|
| dev8-apple | 1 | 13.782 | 56.286 | 0.24× |
| dev8-apple | 96 | 864.203 | 57.154 | 15.12× |
| dev8-remote-cpu | 1 | 31.668 | 88.935 | 0.36× |
| dev8-remote-cpu | 96 | 1816.783 | 91.118 | 19.94× |
| dev8-cuda | 1 | 41.242 | 245.194 | 0.17× |
| dev8-cuda | 96 | 1898.863 | 245.616 | 7.73× |

## Accuracy and release verification

360 timing records. Maximum surface difference from the per-parameter analytical reference is **3.565e-10** in native component units; **0** energy fallbacks in these benchmark cases. Maximum Green adjoint/tangent relative norm difference is **5.650e-14**.

Actual release-wheel suites: Apple 133 tests, 110 passed/23 skipped; CUDA host 133 tests, 124 passed/9 skipped. Checks include Love/Rayleigh modes 0–2, LVZs, recomputed-root finite differences, analytic period derivatives, density scaling, missing roots and critical guards, independent RF tensor physics/attenuation, and Green physical/source/masked/quasistatic derivatives. Skips reflect hardware or optional dependencies. Agreement does not prove root completeness or universally better inversion posteriors.

The release contains Mac CPython 3.12 and Linux CPython 3.13 wheels. Unchanged Metal is reused from dev7 with recorded provenance. No success is claimed for CI jobs that never started. See installation notes and hashed verification records.

## Raw evidence

- [dev8-apple.json](../verification/variational-0.6.0.dev8/dev8-apple.json)
- [dev7-apple.json](../verification/variational-0.6.0.dev8/dev7-apple.json)
- [dev8-cuda.json](../verification/variational-0.6.0.dev8/dev8-cuda.json)
- [dev7-cuda.json](../verification/variational-0.6.0.dev8/dev7-cuda.json)
- [dev8-remote-cpu.json](../verification/variational-0.6.0.dev8/dev8-remote-cpu.json)
- [summary.json](../verification/variational-0.6.0.dev8/summary.json)
- [Critical-coordinate guard](../verification/variational-0.6.0.dev8/critical-guard.json)
- [Measurement protocol](../verification/variational-0.6.0.dev8/README.md)
- [Verification](../verification/preview-0.6.0.dev8/verification.json)
