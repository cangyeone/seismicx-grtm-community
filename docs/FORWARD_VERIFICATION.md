# Forward, gradient, and performance verification (0.3)

English | [简体中文](FORWARD_VERIFICATION.zh-CN.md)

Recorded on 2026-10-06. Version 0.3 introduced NED/SI moment tensors and SDR, streaming/tensor Earth batches, structural Jacobians/VJPs, PyTorch/JAX forward, strain, and stress. See the [API guide](PYTHON_FORWARD.md). This report preserves historical measurements; current packaging results are in [public release verification](../verification/README.md).

## Correctness and coverage

- macOS ARM64, Python 3.14, NumPy 2.5.3, Torch 2.14.1, JAX 0.11.2: 15 passed / 2 GPU skipped out of 17 tests.
- Ubuntu/WSL x86_64, Python 3.8.10, NumPy 1.24.4: C wheel 13 passed / 4 skipped; CUDA wheel 15 passed / 2 optional-framework skipped.
- Independent macOS pip source installation and CPU tests passed.
- Torch backward, shared-source batches, SDR angle gradients, and complex spectral losses agree with NumPy VJP. JAX jit, grad, JVP, jacfwd, vmap, explicit batches, shared-source gradients, and complex losses were checked. CPU framework gradients also agree with independent fixed-grid difference losses.
- CUDA tests cover physical synthesis and structural Jacobians. Framework tests ran on macOS CPU; this report does not claim a complete Torch CUDA tensor or JAX GPU-device test.
- Independent spatial force couples verify all six NED moment components, signs, factors, and normalization, with relative differences below 2×10⁻⁶ for receivers above, below, and across material layers.
- Cartesian receiver differences verify displacement gradients including cylindrical basis-vector derivatives. Strain/stress are symmetric.
- Six moment and three force basis sources satisfy zero free-surface traction: tolerance 2×10⁻¹¹, typical measured values around 10⁻¹⁵.
- Independent central differences check Vp, Vs, density, Qp, Qs, interface depths, deep layers, and direct receiver-layer stress constitutive derivatives.
- SI/native conversion, dimensionless causal histories, batch order/error indices, frozen grids, and interface-topology rejection were checked.
- Native kernel/CLI and six-parameter tangent ownership/workspace ASan/UBSan checks passed.
- Local/remote pip check passed. CUDA runtime linkage is static; ldd shows no libcudart or private build-directory dependency.

## Physical corrections

1. Axisymmetric radial moment terms now use horizontal amplitudes, with matching dz/dr corrections.
2. Upgoing SH order-1 reflection uses a negative sign; downgoing order-2 direct terms use a positive sign, independently verified by force couples.
3. Cross-layer downgoing PSV uses the down-recursion matrix, and receiver reflection propagation uses receiver-layer thickness.
4. Legacy upward vertical force storage is converted to downward NED during physical synthesis; raw `compute()` channel conventions are retained.
5. Strain uses the symmetric Cartesian gradient, including azimuthal/cylindrical basis terms, rather than the historical plotting script's formulas.

These corrections change some historical moment-source outputs. The Fortran timing reference comes from [YunyiQian/grtm](https://github.com/YunyiQian/grtm) and is external to this distribution.

## Portable wheels versus the external reference

Same Threadripper PRO 3955WX (16 cores / 32 threads) and one RTX 5090; one warmup, median of three repetitions. Four-layer near-field model: hs=8 km, hr=0, r=10 km, dt=0.04 s, 2048 samples, PTAM=0, integration period matched to the external reference. All implementations compute force-source displacement bases only. These are integration times; raw JSON also records complete calls.

| Implementation | Integration / s | Relative to external Fortran -O3 |
|---|---:|---:|
| External Fortran -O3 | 11.682756 | 1× |
| 0.3 C wheel, 32 threads | 0.565346 | 20.7× |
| 0.3 CUDA wheel, device 0 | 0.117619 | 99.3× |

Wheels use portable CPU settings (`GRTM_NATIVE=OFF`), unlike earlier tuned native-library benchmarks. The later [public FK report](../benchmarks/fk/README.md) covers other workloads in a separate frozen study. Stress synthesis, framework callbacks, and structural Jacobians are not included in these integration speedups.

## Forward, Earth batches, and Jacobians

Four layers, 64 samples, dt=0.2 s, two distances (10/30 km), moment-source displacement/strain/stress. Complete-call times include Python, allocation, FFT, and physical synthesis.

| Work | CPU / s | CUDA / s |
|---|---:|---:|
| One forward, 32 host threads | 0.005295 | 0.009077 |
| 32 models, serial dispatch | 0.147209 | 0.277019 |
| 32 models, workers=4 / threads=8 | 0.120489 | 0.156355 |

CPU is faster for these small tasks; GPU launch/transfer costs are not amortized. Batches dispatch per model rather than fusing kernels. CPU/CUDA physical relative L2 differences: displacement 4.391×10⁻¹⁴, strain 1.878×10⁻¹⁴, stress 1.904×10⁻¹⁴.

For a complex stress-spectrum Jacobian selecting seven structural parameters and 32 host threads:

| Method | Complete Jacobian / s |
|---|---:|
| Semi-analytic chain derivatives, P tangent integrals | 0.070646 |
| Fixed-grid central differences, 2P solves | 0.076579 |

The semi-analytic method is about 1.08× faster here, not necessarily at every size. Regular-frequency tangents use double; ill-conditioned quasistatic tangents use long double and reuse thread workspaces. Relative L2 difference was 1.435×10⁻⁵ with relative difference step 1e-4. Other benefits include step-independent differentiation, source-basis reuse, and streaming VJP contractions.

## Numerical limits

With PTAM=3, density derivative differences were about 2.51×10⁻⁹; branch switches may still be nonsmooth. An extreme case (four samples, dt=1000 s, r=1000 km) produced analytic/difference density-gradient discrepancies around 0.98 on Linux and 1.00 on macOS ARM64. Structural gradients in this quasistatic, ill-conditioned region **remain unverified**.

Disagreement alone does not establish which method is more accurate: independent higher precision and integration convergence are needed. ARM64 long double adds no precision; the Linux 80-bit path is not a universal guarantee. Use `check_step` and reusable frozen grids to examine such cases.

The raw records for this historical 0.3 report remain in the private development
repository. They are not bundled in this public guide. The separate public
[FK benchmark](../benchmarks/fk/README.md) provides its own numerical data and
scope. See [release verification](../verification/README.md) for installed binary
checks; old timings are not a new measurement of the 0.6 preview.
