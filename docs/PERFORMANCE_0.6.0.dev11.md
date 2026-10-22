# Dispersion energy images: accuracy and speed, 0.6.0.dev11

Measured final installed wheels. These timings compute full forced-response images; they do not measure a speedup of the existing dispersion root search.

Hardware: Threadripper PRO 3955WX 16 cores + RTX 5090 (GPU 1). CPU 1/4 threads; FP64, 64 frequencies × 256 velocities. Includes complex response, raw power, validity and frequency normalization. One warmup, median of three synchronized wall-clock calls; excludes model-object construction and result host copies. Every sampled point is valid.

| Wave | Layers | Batch | CPU 1 thread ms | CPU 4 threads ms | CUDA ms | vs 1 thread | vs 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rayleigh | 8 | 1 | 59.869 | 16.965 | 2.330 | 25.7× | 7.3× |
| Love | 8 | 1 | 22.347 | 5.785 | 1.959 | 11.4× | 3.0× |
| Rayleigh | 8 | 16 | 927.139 | 255.246 | 4.821 | 192.3× | 52.9× |
| Love | 8 | 16 | 331.800 | 93.540 | 2.619 | 126.7× | 35.7× |
| Rayleigh | 32 | 1 | 230.338 | 83.563 | 2.558 | 90.0× | 32.7× |
| Love | 32 | 1 | 73.818 | 22.437 | 2.148 | 34.4× | 10.4× |
| Rayleigh | 32 | 16 | 3825.339 | 1033.854 | 14.419 | 265.3× | 71.7× |
| Love | 32 | 16 | 1285.204 | 354.522 | 5.500 | 233.7× | 64.5× |

Maximum CPU/CUDA complex-response relative L2 difference: **2.246e-13**; maximum normalized-image absolute difference: **1.155e-12**. Independent global P/SV wave-equation checks cover 1/3/8/32 layers, with maximum relative L2 error **4.634e-14**. Tests also cover analytic SH half-space/single-layer solutions, layer splitting, source/density scaling and unit conversion.

12 local peak checks span existing Rayleigh/Love modes 0–2. With η=2e-5 and 4e-5 km/s velocity spacing offset from the roots, the maximum peak/root distance is 1.480e-05 km/s. This is a local-window check, not a universal peak-position bound for other models or the default damping.

Each installed wheel runs 158 regression tests: macOS 126 pass / 32 platform skips; Linux/CUDA 148 pass / 10 skips; no failures. Both wheels pass implementation-source, metadata and checksum audits.

See the [API/physics guide](DISPERSION_IMAGE.md), [raw timing and accuracy records](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/dispersion-image-0.6.0.dev11), and [installed-wheel verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev11).
