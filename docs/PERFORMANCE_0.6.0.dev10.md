# dev10 fused Green batches: accuracy and speed

This compares an unchanged installed dev9 wheel with installed dev10 binaries: legacy reverse AD, legacy wavefield, fused strict and fused fast. Strict preserves the original low-frequency guard; fast admits validated bins to GPU overlap integrals while retaining extreme-frequency protection. CPU, MPS, RF and dispersion algorithms are unchanged.

RTX 5090 / Threadripper PRO 3955WX; Torch threads=1, native host threads=4. FP64, PTAM=0, N=128, K=512 wavenumber steps per frequency, dt=0.4 s, period length=500 km, one receiver. Layers=8/32, B=1/4/16; batch velocities vary by 0.1% per model. All six structural columns and source gradients, squared-waveform objective.

One warmup and three synchronized wall-clock repetitions per method; medians below, all samples retained in JSON. Timings include validation, host metadata transfer, propagation, synthesis, FFT and backward, excluding build/install. Models and integration grids match with no node removal. Strict has 3/64 CPU bins per model; fast has zero CPU bins in this table.

## Forward + all-parameter VJP

| L | B | dev9 AD ms | dev9 wavefield ms | dev10 strict ms | dev10 fast ms | wavefield/strict | wavefield/fast | AD/fast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 441.865 | 215.039 | 212.027 | 32.611 | 1.01× | 6.59× | 13.55× |
| 8 | 4 | 1861.536 | 932.469 | 894.016 | 100.818 | 1.04× | 9.25× | 18.46× |
| 8 | 16 | 7307.377 | 4126.933 | 3516.751 | 365.433 | 1.17× | 11.29× | 20.00× |
| 32 | 1 | 3539.108 | 969.595 | 856.143 | 79.257 | 1.13× | 12.23× | 44.65× |
| 32 | 4 | 14462.191 | 3433.517 | 3505.059 | 272.950 | 0.98× | 12.58× | 52.98× |
| 32 | 16 | 57365.357 | 14337.333 | 13215.841 | 1028.001 | 1.08× | 13.95× | 55.80× |

## Forward only

| L | B | dev9 AD ms | dev9 wavefield ms | dev10 strict ms | dev10 fast ms | wavefield/strict | wavefield/fast | AD/fast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 52.841 | 52.619 | 53.446 | 7.215 | 0.98× | 7.29× | 7.32× |
| 8 | 4 | 208.739 | 236.031 | 196.078 | 15.365 | 1.20× | 15.36× | 13.59× |
| 8 | 16 | 838.900 | 942.335 | 766.149 | 47.696 | 1.23× | 19.76× | 17.59× |
| 32 | 1 | 190.632 | 200.500 | 193.479 | 19.365 | 1.04× | 10.35× | 9.84× |
| 32 | 4 | 769.218 | 768.468 | 742.741 | 52.070 | 1.03× | 14.76× | 14.77× |
| 32 | 16 | 3391.038 | 3373.196 | 2952.635 | 167.781 | 1.14× | 20.10× | 20.21× |

## Isolating cross-model fusion

| L | B | serial fast ms | fused fast ms | serial/fused |
|---:|---:|---:|---:|---:|
| 8 | 16 | 498.039 | 365.433 | 1.36× |
| 32 | 16 | 1251.065 | 1028.001 | 1.22× |

Both columns use dev10 fast, changing only `cuda_batch`. This isolates fusion; wavefield/fast above also includes the low-frequency path change. A single model already exposes many frequency/wavenumber tasks, so fusion does not imply another B-fold speedup.

## Same-host CPU reference

| L | B | CPU wavefield ms | CUDA fast ms | CPU/CUDA |
|---:|---:|---:|---:|---:|
| 8 | 1 | 823.015 | 32.611 | 25.24× |
| 32 | 1 | 2851.363 | 79.257 | 35.98× |

The CPU reference is unchanged dev9 with four native threads. This release claims no additional pure-CPU speedup. CPU/CUDA combines device and guard-policy differences.

## Accuracy and coverage

Maximum relative norm errors across records: waveform **1.263e-08**, model gradient **1.921e-08**, source gradient **1.183e-08**, individual parameter column **1.639e-07**.

Another 48 accuracy records cover 8/32 layers, B=4, ordinary gradients, LVL, strong gradients, very thin layers, shallow half-spaces and low Q, with ordinary/long windows and fixed random linear waveform cotangents. Strict/fast compare against original reverse AD on the same model. Long windows retain protection; ordinary-window speedups do not transfer automatically. Thin or special geometry may use GPU AD fallback. All records meet waveform 1e-4 and model/source/per-column gradient 1e-3 targets.

These are finite-case numerical audits, not universal error bounds, proof of continuous-integral convergence or superior inversion accuracy. Strict remains the default. Small timing differences are sensitive to three-sample/system noise. Model/grid planning remains host work; only zero CPU frequency counts indicate fully GPU numerical propagation/gradients.

Each installed wheel runs 149 tests: Apple 118 passed/31 platform skips; Linux/CUDA 140 passed/9 platform skips. Coverage includes varying topology, physical fields, units, masks, nondefault streams, retained backward and runtime teardown.

[API](CUDA_GREEN_BATCH.md) · [Install](PREVIEW_0.6.0.dev10.md) · [Verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev10)

Raw records: [legacy-cuda.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/legacy-cuda.json) · [fused-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/fused-final.json) · [serial-fast-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/serial-fast-final.json) · [validation-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/validation-final.json) · [cpu-reference.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/cpu-reference.json)
