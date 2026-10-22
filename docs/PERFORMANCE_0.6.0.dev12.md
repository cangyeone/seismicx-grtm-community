# Dispersion-image adjoints: accuracy and performance (0.6.0.dev12)

RTX 5090 (CUDA:1) versus Threadripper PRO 3955WX, native CPU 4 threads and Torch operators 1 thread. FP64, 64 frequencies × 256 velocities, damping=0.002. Synchronized full forward + power objective + all-layer-parameter VJP, one warmup and median of three. Model construction and final host copy excluded. Array bases use a warm cache; first-time basis construction is excluded.

| Layers | Batch | Wave / observation | CPU (ms) | CUDA (ms) | Speedup |
|---:|---:|---|---:|---:|---:|
| 8 | 1 | rayleigh direct | 169.196 | 4.051 | 41.8× |
| 8 | 1 | love direct | 13.481 | 3.088 | 4.4× |
| 8 | 16 | rayleigh direct | 1844.034 | 22.341 | 82.5× |
| 8 | 16 | love direct | 196.287 | 5.089 | 38.6× |
| 32 | 1 | rayleigh direct | 442.244 | 8.656 | 51.1× |
| 32 | 1 | love direct | 52.282 | 3.638 | 14.4× |
| 32 | 16 | rayleigh direct | 7153.278 | 79.276 | 90.2× |
| 32 | 16 | love direct | 762.933 | 14.398 | 53.0× |
| 32 | 1 | rayleigh fk coherent | 1082.719 | 17.407 | 62.2× |
| 32 | 1 | rayleigh fj coherent | 532.646 | 11.286 | 47.2× |
| 32 | 1 | rayleigh fj noise | 445.406 | 10.172 | 43.8× |
| 32 | 16 | rayleigh fk coherent | 15038.916 | 158.710 | 94.8× |
| 32 | 16 | rayleigh fj coherent | 7077.971 | 81.906 | 86.4× |
| 32 | 16 | rayleigh fj noise | 7208.355 | 81.023 | 89.0× |

Array timing uses Rayleigh, 32 receiver distances, 4 km length, 0.2 km offset, rectangular window and the display grid as integration mesh. This workload is not a spatial-convergence certificate. Maximum CUDA/CPU gradient relative L2 difference: **3.46e-12**.

Independent generic-matrix-solve PyTorch GRTM reverse-AD comparison: 16×64 grid, one CPU thread for both implementations, same GPU for CUDA. Ratios below are PyTorch AD time / explicit-adjoint time, not a claim against every optimized AD implementation or an earlier production version.

| Layers | Batch | Wave | CPU ratio | CUDA ratio |
|---:|---:|---|---:|---:|
| 8 | 1 | rayleigh | 1.14× | 14.26× |
| 8 | 1 | love | 2.85× | 8.21× |
| 8 | 16 | rayleigh | 0.57× | 7.98× |
| 8 | 16 | love | 0.67× | 3.53× |
| 32 | 1 | rayleigh | 1.12× | 29.35× |
| 32 | 1 | love | 3.10× | 26.03× |
| 32 | 16 | rayleigh | 0.57× | 22.04× |
| 32 | 16 | love | 0.64× | 12.71× |

CPU does not improve every workload: at one thread, batch=16 explicit adjoints achieve 0.57–0.67× the reference speed, taking about 1.5–1.8× as long. CUDA gains are larger. These slower CPU cases are retained.

Maximum gradient relative L2 difference against independent AD: **3.97e-12**.

Independent global P/SV wave-equation validation covers 1/3/8/32 layers. Directional finite differences are used only for validation; at step 1e-5 their relative discrepancies from the adjoint are of order 1e-11–1e-8. Tests also check complex-source gradients, PSD linearity/normalization degeneracy, physical negative-k parity, independent angular averaging of J0 covariance, SI conversion, grid refinement, NaN-data masks and nondefault CUDA streams.

Two-parameter noiseless local inversions recover surface Vs and first-interface depth for both active and noise FJ, with all other parameters/source fixed. This is a usability check, not proof of global convergence or statistical superiority.

Final installed binary tests: macos 136 passed / 33 platform skips (169 total), linux 159 passed / 10 platform skips (169 total).

Raw numerical evidence, individual timing samples and comparison figures: [verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/dispersion-adjoint-0.6.0.dev12). [API and physical definitions](DISPERSION_IMAGE.md).
