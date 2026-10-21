# Analytic adjoints and device differentiation — 0.6.0.dev7

[Methods and APIs](ANALYTIC_METHODS.md) · [Device boundaries](DEVICE_EXECUTION.md) · [Binary installation](PREVIEW_0.6.0.dev7.md)

Installed compiled dev6 and dev7 wheels are compared on Apple M4 Max CPU, Threadripper PRO 3955WX CPU and RTX 5090. **Speedup means same-device dev6 time / dev7 time**; below 1 means slower. CPU/GPU ratios are reported separately.

## Timing scope

FP64 throughout; two warmups and synchronized wall-time medians of five calls, or three for larger Green cases. Four native CPU threads and one Torch thread. Versions run serially without overlapping compilation or other computation from this task; external shared-machine load is not excluded.

Surface results include root finding and all thickness/Vp/Vs/density kernels at eight periods; tables use 32 layers. RF uses 32 layers, two rays and 256 FFT points, including complete forward or loss VJP. Small Green uses eight layers, two receivers, 32 time samples, 16 frequencies, 64 fixed wavenumber steps and all structural parameters. Green batches still dispatch individual models. A fixed grid makes algorithm comparisons fair but does not certify physical cutoff convergence.

## Apple M4 Max CPU

| Operation | B | dev6 ms | dev7 ms | Speedup |
|---|---:|---:|---:|---:|
| surface Phase + all kernels | 1 | 11.135 | 2.820 | 3.95× |
| surface Group + all kernels | 1 | 45.149 | 24.252 | 1.86× |
| rf Forward only | 1 | 0.457 | 0.471 | 0.97× |
| rf Forward + VJP | 1 | 1.785 | 1.268 | 1.41× |
| green Forward only | 1 | 1.204 | 1.221 | 0.99× |
| green Forward + VJP | 1 | 14.187 | 14.073 | 1.01× |
| surface Phase + all kernels | 32 | 305.028 | 43.468 | 7.02× |
| surface Group + all kernels | 32 | 1393.913 | 728.848 | 1.91× |
| rf Forward only | 32 | 7.846 | 7.978 | 0.98× |
| rf Forward + VJP | 32 | 28.488 | 27.386 | 1.04× |
| green Forward only | 32 | 35.722 | 35.088 | 1.02× |
| green Forward + VJP | 32 | 451.048 | 462.044 | 0.98× |

## Threadripper PRO 3955WX CPU

| Operation | B | dev6 ms | dev7 ms | Speedup |
|---|---:|---:|---:|---:|
| surface Phase + all kernels | 1 | 23.877 | 9.733 | 2.45× |
| surface Group + all kernels | 1 | 105.440 | 60.153 | 1.75× |
| rf Forward only | 1 | 1.489 | 1.372 | 1.09× |
| rf Forward + VJP | 1 | 6.363 | 4.464 | 1.43× |
| green Forward only | 1 | 2.636 | 2.880 | 0.92× |
| green Forward + VJP | 1 | 18.884 | 19.431 | 0.97× |
| surface Phase + all kernels | 32 | 539.796 | 108.964 | 4.95× |
| surface Group + all kernels | 32 | 3257.861 | 1688.588 | 1.93× |
| rf Forward only | 32 | 18.142 | 18.993 | 0.96× |
| rf Forward + VJP | 32 | 60.095 | 58.043 | 1.04× |
| green Forward only | 32 | 86.825 | 87.743 | 0.99× |
| green Forward + VJP | 32 | 641.002 | 634.238 | 1.01× |

## RTX 5090 CUDA

| Operation | B | dev6 ms | dev7 ms | Speedup |
|---|---:|---:|---:|---:|
| surface Phase + all kernels | 1 | 11.256 | 12.365 | 0.91× |
| surface Group + all kernels | 1 | 43.899 | 54.015 | 0.81× |
| rf Forward only | 1 | 1.944 | 1.861 | 1.04× |
| rf Forward + VJP | 1 | 12.093 | 5.930 | 2.04× |
| green Forward only | 1 | 2.502 | 4.707 | 0.53× |
| green Forward + VJP | 1 | 118.257 | 48.042 | 2.46× |
| surface Phase + all kernels | 32 | 17.723 | 14.847 | 1.19× |
| surface Group + all kernels | 32 | 69.901 | 56.851 | 1.23× |
| rf Forward only | 32 | 1.871 | 1.878 | 1.00× |
| rf Forward + VJP | 32 | 11.797 | 5.970 | 1.98× |
| green Forward only | 32 | 73.828 | 146.912 | 0.50× |
| green Forward + VJP | 32 | 3765.435 | 1462.136 | 2.58× |

## Larger Green grid

Eight layers, 128 time samples, 64 frequencies, 512 fixed wavenumber steps and two receivers.

| Device | Operation | B | dev6 ms | dev7 ms | Speedup |
|---|---|---:|---:|---:|---:|
| cpu | Forward only | 1 | 75.507 | 68.236 | 1.11× |
| cpu | Forward + VJP | 1 | 631.076 | 687.425 | 0.92× |
| cuda:1 | Forward only | 1 | 20.507 | 36.146 | 0.57× |
| cuda:1 | Forward + VJP | 1 | 3717.588 | 453.250 | 8.20× |
| cpu | Forward only | 4 | 277.501 | 273.830 | 1.01× |
| cpu | Forward + VJP | 4 | 2559.849 | 2760.621 | 0.93× |
| cuda:1 | Forward only | 4 | 79.694 | 146.436 | 0.54× |
| cuda:1 | Forward + VJP | 4 | 14892.448 | 1810.759 | 8.22× |

## Same-host dev7 CPU / CUDA

| Operation | B | CPU / CUDA |
|---|---:|---:|
| surface Phase + all kernels | 1 | 0.79× |
| surface Group + all kernels | 1 | 1.11× |
| rf Forward only | 1 | 0.74× |
| rf Forward + VJP | 1 | 0.75× |
| green Forward only | 1 | 0.61× |
| green Forward + VJP | 1 | 0.40× |
| surface Phase + all kernels | 32 | 7.34× |
| surface Group + all kernels | 32 | 29.70× |
| rf Forward only | 32 | 10.11× |
| rf Forward + VJP | 32 | 9.72× |
| green Forward only | 32 | 0.60× |
| green Forward + VJP | 32 | 0.43× |

## Accuracy and limits

The normalized maximum difference is `max(abs(new-old))/max(abs(old))`, not a componentwise relative error near zero or an inversion-accuracy claim. The table covers all reported models/devices/batches.

| Family | Maximum normalized difference |
|---|---:|
| surface | 9.562e-11 |
| rf | 5.280e-15 |
| green | 3.028e-12 |

Maximum normalized CPU/CUDA difference on the same host, including the larger grid:

| Family | dev6 | dev7 |
|---|---:|---:|
| surface | 1.738e-07 | 1.738e-07 |
| rf | 3.738e-15 | 4.953e-15 |
| green | 2.945e-12 | 4.271e-13 |

Independent tests additionally cover root-recomputed higher-mode differences and the original analytic kernels, an independent RF Torch graph, Levin waveform-JVP density identities and Gauss differences, and all Green physical fields/nondefault CUDA streams/direct stress terms/shared-source batches/gradient masks. Long-period checks tightened the shared low-frequency guard from 3e-3 to 5e-3 across forward/JVP/adjoint. FP64 and root tolerances were not relaxed. See [binary release verification](../verification/performance-0.6.0.dev7/verification.json) for the complete test evidence.

**Speed does not improve everywhere.** CUDA launch/synchronization and Torch synthesis overhead can dominate small tasks; `device_pipeline=False` retains the Green host-synthesis reference. Surface `differentiation="forward"` retains the original analytic kernels and can be faster for small batches. Critical group-velocity coordinates use the protected original path, so ideal O(L) cost or one noncritical root cannot predict complete calls. CPU Green structural algorithms are largely unchanged; their timing differences are not evidence of an adjoint algorithm gain.

Adaptive Levin remains CPU-only; the new CUDA reverse path uses a fixed grid. Levin applies to Green oscillatory wavenumber integrals, not RF FFT or dispersion roots. Prior [Levin/PTAM accuracy and timing](QUADRATURE_0.6.0.dev6.md) and [disba/pyfk comparisons](PERFORMANCE_0.6.0.dev5.md) remain historical evidence and were not remeasured here; higher-mode disba kernels are not ground truth. MPS RF/surface remains a CPU FP64 bridge.

## Inspectable evidence

- [summary.json](../verification/performance-0.6.0.dev7/benchmarks/summary.json)
- [before-apple-final.json](../verification/performance-0.6.0.dev7/benchmarks/before-apple-final.json)
- [after-apple-final.json](../verification/performance-0.6.0.dev7/benchmarks/after-apple-final.json)
- [before-cuda-final.json](../verification/performance-0.6.0.dev7/benchmarks/before-cuda-final.json)
- [after-cuda-final.json](../verification/performance-0.6.0.dev7/benchmarks/after-cuda-final.json)
- [before-cuda-large.json](../verification/performance-0.6.0.dev7/benchmarks/before-cuda-large.json)
- [after-cuda-large.json](../verification/performance-0.6.0.dev7/benchmarks/after-cuda-large.json)

Matching NPZ archives preserve complete outputs; JSON stores every timing and SHA-256. The source repository provides `optimization/analytic-adjoint-20261021/measure.py` to reproduce configurations.
