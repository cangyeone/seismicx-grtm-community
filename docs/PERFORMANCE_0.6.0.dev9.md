# dev9 explicit Green integrals: accuracy and speed

The opt-in `vjp_method="wavefield"` evaluates closed-form forward/adjoint layer overlaps using the existing GRTM modes. CPU/CUDA share analytical formulas; there is no depth quadrature or per-parameter propagation perturbation. Default method selection is unchanged.

## Measurement scope

These are installed dev9 binary measurements, solver source `f3d8106fc059ab75ba5580fbaa3fadcaeb908917`. Both methods run in the same wheel with identical FP64 models, fixed wavenumber grids, PTAM=0 and scalar objectives. All six model columns and source gradients are selected (6L−1 active structural parameters). Batch velocities vary by 0.1% per member.

Two warmups, three synchronized wall-time repetitions; medians below, every sample retained. Forward+VJP includes propagation, physical synthesis, FFT and backward. VJP-only retains the graph but both native paths reconstruct wavefields. Native CPU threads=4 and Torch threads=1 throughout, including protected CUDA host work.

N=16/K=32 uses 8 nonnegative frequency bins; N=128/K=512 uses 64. K denotes wavenumber steps per frequency. Large P=0 uses dt=0.1 s / period length 1000 km, with no protected bins. Large P=3 uses dt=0.4 s / 500 km, retaining 3/64 protected bins. Small grids use dt=0.4 s / 500 km with P=0. Large P=0 and P=3 are different workloads, not an accuracy relaxation.

The 144 final timing records have a maximum model-gradient relative norm difference of **2.067e-14** against reverse AD. Additional tests cover all 30 raw channels, force/moment sources, every material column, geometry, physical stress/strain, nondefault streams, central differences and CPU/CUDA agreement. This measures derivative consistency, not convergence to the continuous wavenumber integral or universally better inversions.

## Same-device comparison against reverse AD

All times are ms. Ratios are AD/wavefield; only values above 1 represent speedups. B is the number of models.

### Apple M4 Max CPU

| L | N/K | B | P | AD forward+VJP | Wavefield forward+VJP | Ratio | AD VJP | Wavefield VJP | VJP ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16/32 | 1 | 0 | 4.592 | 3.302 | 1.39× | 3.787 | 2.622 | 1.44× |
| 8 | 16/32 | 4 | 0 | 17.477 | 12.767 | 1.37× | 15.029 | 10.385 | 1.45× |
| 8 | 16/32 | 16 | 0 | 69.354 | 52.122 | 1.33× | 59.661 | 41.793 | 1.43× |
| 32 | 16/32 | 1 | 0 | 13.808 | 9.149 | 1.51× | 12.636 | 7.977 | 1.58× |
| 32 | 16/32 | 4 | 0 | 54.998 | 36.305 | 1.51× | 50.375 | 31.785 | 1.58× |
| 32 | 16/32 | 16 | 0 | 217.916 | 146.541 | 1.49× | 201.010 | 128.476 | 1.56× |
| 8 | 128/512 | 1 | 3 | 411.774 | 283.494 | 1.45× | 389.902 | 259.327 | 1.50× |
| 8 | 128/512 | 4 | 3 | 1651.172 | 1131.118 | 1.46× | 1569.930 | 1047.337 | 1.50× |
| 32 | 128/512 | 1 | 3 | 1574.906 | 1055.164 | 1.49× | 1496.129 | 969.101 | 1.54× |
| 8 | 128/512 | 1 | 0 | 415.997 | 265.714 | 1.57× | 393.478 | 243.806 | 1.61× |
| 8 | 128/512 | 4 | 0 | 1653.668 | 1064.038 | 1.55× | 1575.171 | 971.410 | 1.62× |
| 32 | 128/512 | 1 | 0 | 1574.600 | 987.284 | 1.59× | 1495.378 | 904.295 | 1.65× |

### Threadripper PRO 3955WX CPU

| L | N/K | B | P | AD forward+VJP | Wavefield forward+VJP | Ratio | AD VJP | Wavefield VJP | VJP ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16/32 | 1 | 0 | 9.134 | 8.304 | 1.10× | 7.287 | 6.687 | 1.09× |
| 8 | 16/32 | 4 | 0 | 37.358 | 35.778 | 1.04× | 32.922 | 31.919 | 1.03× |
| 8 | 16/32 | 16 | 0 | 146.392 | 145.442 | 1.01× | 123.381 | 112.538 | 1.10× |
| 32 | 16/32 | 1 | 0 | 26.745 | 33.645 | 0.79× | 23.587 | 30.158 | 0.78× |
| 32 | 16/32 | 4 | 0 | 111.783 | 100.754 | 1.11× | 93.963 | 88.681 | 1.06× |
| 32 | 16/32 | 16 | 0 | 426.496 | 389.428 | 1.10× | 382.470 | 344.207 | 1.11× |
| 8 | 128/512 | 1 | 3 | 763.951 | 770.074 | 0.99× | 686.813 | 672.124 | 1.02× |
| 8 | 128/512 | 4 | 3 | 3100.473 | 3251.855 | 0.95× | 3013.690 | 2739.802 | 1.10× |
| 32 | 128/512 | 1 | 3 | 2841.005 | 3061.671 | 0.93× | 2626.427 | 2579.700 | 1.02× |
| 8 | 128/512 | 1 | 0 | 692.446 | 595.180 | 1.16× | 632.026 | 675.188 | 0.94× |
| 8 | 128/512 | 4 | 0 | 2819.039 | 2707.541 | 1.04× | 2838.400 | 2516.462 | 1.13× |
| 32 | 128/512 | 1 | 0 | 2568.050 | 2446.894 | 1.05× | 2350.353 | 2111.667 | 1.11× |

### RTX 5090 CUDA

| L | N/K | B | P | AD forward+VJP | Wavefield forward+VJP | Ratio | AD VJP | Wavefield VJP | VJP ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16/32 | 1 | 0 | 17.996 | 10.580 | 1.70× | 12.223 | 5.173 | 2.36× |
| 8 | 16/32 | 4 | 0 | 69.252 | 40.722 | 1.70× | 48.026 | 20.108 | 2.39× |
| 8 | 16/32 | 16 | 0 | 273.481 | 162.139 | 1.69× | 195.168 | 79.999 | 2.44× |
| 32 | 16/32 | 1 | 0 | 63.631 | 16.232 | 3.92× | 55.850 | 8.002 | 6.98× |
| 32 | 16/32 | 4 | 0 | 253.159 | 57.674 | 4.39× | 221.680 | 27.097 | 8.18× |
| 32 | 16/32 | 16 | 0 | 1003.943 | 223.737 | 4.49× | 888.661 | 106.910 | 8.31× |
| 8 | 128/512 | 1 | 3 | 443.404 | 217.374 | 2.04× | 390.151 | 163.255 | 2.39× |
| 8 | 128/512 | 4 | 3 | 1777.484 | 952.146 | 1.87× | 1718.089 | 789.910 | 2.18× |
| 32 | 128/512 | 1 | 3 | 3536.411 | 900.864 | 3.93× | 3373.328 | 816.583 | 4.13× |
| 8 | 128/512 | 1 | 0 | 258.021 | 32.149 | 8.03× | 249.822 | 23.965 | 10.42× |
| 8 | 128/512 | 4 | 0 | 1013.935 | 121.688 | 8.33× | 981.465 | 93.629 | 10.48× |
| 32 | 128/512 | 1 | 0 | 3047.155 | 93.416 | 32.62× | 3026.502 | 73.828 | 40.99× |

## Same-host CPU/CUDA: both use wavefield kernels

| L | N/K | B | P | CPU ms | CUDA ms | CPU/CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16/32 | 1 | 0 | 8.304 | 10.580 | 0.78× |
| 8 | 16/32 | 4 | 0 | 35.778 | 40.722 | 0.88× |
| 8 | 16/32 | 16 | 0 | 145.442 | 162.139 | 0.90× |
| 32 | 16/32 | 1 | 0 | 33.645 | 16.232 | 2.07× |
| 32 | 16/32 | 4 | 0 | 100.754 | 57.674 | 1.75× |
| 32 | 16/32 | 16 | 0 | 389.428 | 223.737 | 1.74× |
| 8 | 128/512 | 1 | 3 | 770.074 | 217.374 | 3.54× |
| 8 | 128/512 | 4 | 3 | 3251.855 | 952.146 | 3.42× |
| 32 | 128/512 | 1 | 3 | 3061.671 | 900.864 | 3.40× |
| 8 | 128/512 | 1 | 0 | 595.180 | 32.149 | 18.51× |
| 8 | 128/512 | 4 | 0 | 2707.541 | 121.688 | 22.25× |
| 32 | 128/512 | 1 | 0 | 2446.894 | 93.416 | 26.19× |

## QEDispInv: phase kernels at known roots

| L | B | QE original ms | QE full FP64 ms | GRTM old path ms | GRTM energy ms | Own speedup | QE original max abs diff | QE full FP64 max abs diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.1051 | 0.0750 | 0.2641 | 0.1625 | 1.63× | 3.79e-08 | 1.63e-13 |
| 8 | 32 | 1.5236 | 1.2035 | 3.2371 | 2.3102 | 1.40× | 3.79e-08 | 1.63e-13 |
| 32 | 1 | 0.1392 | 0.1411 | 0.4414 | 0.2833 | 1.56× | 5.50e-08 | 2.21e-13 |
| 32 | 32 | 4.4236 | 4.4364 | 14.0386 | 9.0996 | 1.54× | 5.50e-08 | 2.21e-13 |
| 96 | 1 | 0.4159 | 0.4248 | 1.3560 | 0.8623 | 1.57× | 5.12e-08 | 4.43e-14 |
| 96 | 32 | 13.4772 | 13.4395 | 43.6393 | 27.3394 | 1.60× | 5.12e-08 | 4.43e-14 |

Eight periods, fundamental Rayleigh mode, one CPU thread and identical FP64 roots. QE receives known phase velocity; root searches and upstream C++ map overhead are excluded. GRTM likewise times only energy kernels. These are not full dispersion-forward or Green wavenumber-integration comparisons.

QE benefits from specialized real CPS formulas and a smaller wrapper. Its input model is float32 but much of the internal arithmetic is already double. Input promotion alone leaves SNGL conversions and default-precision constants. Promoting all of these reduces the largest difference here to about 2.2e−13. This fundamental-mode experiment does not certify higher-mode identification near cutoffs.

Removing unused Fc/Ft preparation improves our 32-layer B=32 phase-energy kernel by about 1.54×. Original QE remains about 2.06× faster. No QE implementation was copied into our solver. Period derivatives needed by group kernels are retained.

## CUDA gains and limits

Layer modes and overlaps run over (frequency,wavenumber,layer), block solves over (frequency,wavenumber), followed by deterministic reduction without an AD tape on regular bins. FFT, synthesis and regular gradients remain on GPU. Small jobs retain launch/synchronization overhead; long windows retain the original extended-precision CPU protection for low frequencies. Its threshold is unchanged. Batches remain per-model native calls, not fused cross-model launches.

Only the derivative path changes, so unchanged forward cost limits end-to-end gains. VJP-only speedups can be larger. Three repetitions quantify these workloads, not hardware-independent guarantees; very small timings are sensitive to system noise. No precision reduction, node removal or omitted source/interface terms is used.

## Verification and evidence

Each installed wheel runs 143 tests: Apple 117 passed/26 platform skips; Linux/CUDA 134 passed/9 platform skips. Logs, packaging audits and hashes are published. CUDA subprocess regression tests also verify autograd worker teardown after runtime shutdown.

CPU gains are not universal: M4 Max end-to-end ratios are 1.33–1.59×, while Threadripper ratios are 0.79–1.16×, including slowdowns. The default remains unchanged; choose the CPU path for the actual workload rather than extrapolating GPU gains.

## GPU activity diagnosis (separate profiling)

| L | N/K | P | AD GPU activity ms | Wavefield GPU activity ms | Ratio |
|---:|---:|---:|---:|---:|---:|
| 32 | 16/32 | 0 | 51.376 | 3.089 | 16.63× |
| 8 | 128/512 | 3 | 247.681 | 19.004 | 13.03× |
| 8 | 128/512 | 0 | 244.939 | 19.390 | 12.63× |

These are sums of CUDA activity durations in separate VJP profiles, excluding CPU protection and launch gaps, not end-to-end speedups. First-use profiler initialization wall time is excluded from benchmark results. For the 32-layer small grid, the old reverse-node kernel takes about 51.30 ms; new block solves/layer overlaps/parameter mapping take about 1.46/0.65/0.64 ms. The 8-layer large-grid GPU activity falls to about 19 ms, while low-frequency protection can still leave a 163 ms full VJP.

[Methods/API](GREEN_WAVEFIELD_KERNELS.md) · [Release/ABI](PREVIEW_0.6.0.dev9.md) · [Verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev9)

[QEDispInv kernel entry](https://github.com/pan3rock/QEDispInv/blob/ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa/src/swegn96.cc) · [Fortran reference](https://github.com/pan3rock/QEDispInv/blob/ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa/fortran/sregn96.f90)

Raw records: [apple-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/apple-final.json) · [apple-regular-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/apple-regular-final.json) · [cpu-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/cpu-final.json) · [cpu-regular-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/cpu-regular-final.json) · [cuda-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/cuda-final.json) · [cuda-regular-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/cuda-regular-final.json) · [qe-known-root-apple.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/qe-known-root-apple.json) · [cuda-profile.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-wavefield-0.6.0.dev9/cuda-profile.json)
