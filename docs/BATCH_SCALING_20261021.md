# Model-batch scaling: latency and throughput — 2026-10-21

English | [简体中文](BATCH_SCALING_20261021.zh-CN.md)

Batching helps RF and surface-wave CUDA throughput. Green functions use a
different execution path: a host pool schedules separate model calls, and
structural VJP remains CPU. Increasing its batch does not create a fused GPU
adjoint. These are fresh installed **0.6.0.dev4** measurements with unchanged
FP64/complex128, physics and the 1e-10 km/s root tolerance. No new binary is needed.

## CUDA speedup over CPU at the same batch size

Each entry is **CPU batch time / CUDA batch time**, not a comparison of different
model counts. RF baseline: 4 CPU Torch threads (1 thread is also recorded in
the CSV). Surface baseline: 16 model workers. Green baseline: a total budget of
16 CPU threads; B1 uses 1 model worker × 16 native threads, B16/B64 use 4 × 4.

| Task and CPU baseline | B=1 | B=16 | B=64 | B=256 |
| --- | --- | --- | --- | --- |
| RF (CPU4) forward | 11.69× | 41.01× | 86.75× | 81.75× |
| RF (CPU4) forward + VJP | 7.57× | 24.29× | 13.87× | 27.25× |
| Phase (CPU16) forward | 0.28× | 0.35× | 0.84× | 1.50× |
| Phase (CPU16) + full Vs kernel | 1.23× | 1.72× | 3.11× | 4.40× |
| Group (CPU16) forward | 0.19× | 0.42× | 0.86× | 1.55× |
| Group (CPU16) + full Vs kernel | 2.87× | 3.66× | 5.88× | 6.67× |
| Green (CPU16) † forward | 1.14× | 1.41× | 1.93× | — |
| Green (CPU16) † forward + VJP | 1.02× | 1.04× | — | — |

Values below 1 are slower. † Green CUDA rows include CPU preparation and host
input/output. Its VJP is CPU work in both columns. B64 Green VJP and B256 Green
were not measured; dashes are not estimates. The previous headline table used
B1 Green but B16 RF/surface, so it did not show scaling across batches.

## GPU throughput and gain over its own B1 throughput

Units: **models/s**. Gain is `(B/T_B)/(1/T_1) = B*T_1/T_B`, with each `T_B` a
measured whole-batch median. This differs from GPU-versus-CPU acceleration.

| Task | B=1 | B=16 | B=64 | B=256 | Largest measured batch / B1 |
| --- | --- | --- | --- | --- | --- |
| RF forward | 480.8 | 7157.2 | 17199.8 | 17755.9 | 36.93× (B=256) |
| RF forward + VJP | 85.5 | 1353.5 | 981.8 | 1857.8 | 21.73× (B=256) |
| Phase forward | 83.6 | 1225.5 | 3326.3 | 6475.7 | 77.43× (B=256) |
| Phase + full Vs kernel | 51.3 | 700.0 | 1482.5 | 2130.8 | 41.50× (B=256) |
| Group forward | 73.4 | 1090.8 | 3035.1 | 6178.5 | 84.21× (B=256) |
| Group + full Vs kernel | 19.4 | 241.4 | 429.3 | 539.6 | 27.87× (B=256) |
| Green † forward | 11.5 | 27.1 | 38.5 | — | 3.34× (B=64) |
| Green † forward + VJP | 1.4 | 2.2 | — | — | 1.64× (B=16) |

Green throughput gain also reflects reallocation of the same CPU thread budget
and host concurrency; it is not native CUDA model fusion. RF B64 gradient
throughput is below B16 in this run: larger batches are not monotonically faster
per model. Both forward and backward have distinct saturation/memory costs.

## Why some speedups remain modest

RF and surface-wave CUDA kernels have model axes; enough independent models
amortize validation, dispatch and synchronization and expose more GPU work.
Pure surface-wave forward is relatively small. CPU fundamental-mode continuation
can reuse previous periods to reduce discovery, while CUDA currently evaluates
more trial points in parallel for each period. Native CUDA full kernels have
more parameter work to amortize their overhead. Green has no fused model axis
in its CUDA entry point, allocates/transfers per model and performs structural
AD on CPU; outer batching cannot remove that boundary.

The old campaign had other GPU work around 80% utilization. This sweep started
with sampled utilization 0%, and records GPU states before/after each case
(samples can include this benchmark itself). It is still shared hardware.
Changes from yesterday's absolute times must not all be attributed to batching;
all scaling ratios above use only this new sweep.

## Protocol and numerical checks

Threadripper PRO 3955WX, RTX 5090 32 GB (`cuda:1`), Python 3.13.5,
Torch 2.8.0+cu128. 32 layers; the same fixed cohort and prefix subsets at
B=1/16/64/256. One warmup, three synchronized host-wall-time trials, rotating
backends and operations; medians. Public validation included. Prepared RF and
surface operators use resident models/cached search meshes; initial construction
and transfer are excluded. Green host I/O and transfer are included because that
is its existing interface. The recorded peak allocation is a Torch counter;
it excludes native Green CUDA malloc. No other users' work was stopped.

RF: 1 ray, 6 Gaussian bands, nfft=2048, 81 time points and 41 polarization windows.
Its derivative is a scalar VJP of fixed RF/apparent-Vs cotangents w.r.t.
top/Vp/Vs/density, with the first top fixed. Surface: Rayleigh fundamental,
12 periods 2–50 s, full all-layer Vs kernels including roots. Green: 4 receivers,
64 time samples, fixed 20205-node grid; Vp/Vs VJP including the base forward.
No dense waveform Jacobian is claimed.

All 34 backend/batch records passed CPU/CUDA numerical comparisons:

| Family | Max relative forward difference | Max relative derivative difference |
| --- | --- | --- |
| receiver_function  | 1.159e-15 | 2.634e-15 |
| dispersion phase | 5.194e-13 | 3.168e-11 |
| dispersion group | 1.808e-11 | 2.740e-09 |
| green  | 2.442e-14 | 0.000e+00 |

Separate checks compare native RF batch16 with individual calls; surface
batch256 thread-based roots with single-model warp-based roots; and concurrent
Green CPU adjoints with serial CPU adjoints at representative models:

| Family | Metric | Maximum |
| --- | --- | --- |
| receiver_function | forward_relative_error | 0.000e+00 |
| receiver_function | vjp_relative_error | 1.346e-15 |
| dispersion | forward_relative_error | 2.239e-11 |
| dispersion | kernel_relative_error | 2.936e-09 |
| green | vjp_relative_error | 4.309e-19 |

These are observed numerical checks, not universal error bounds. Original
70-digit accuracy and unchanged-disba comparisons remain in the
[dev4 report](PERFORMANCE_0.6.0.dev4.md); do not mix its separate campaigns into
the new throughput ratios.

## Reproduction and data

```sh
python benchmarks/forward-derivatives/benchmark_batch_scaling.py --device cuda:1 --layers 32 --batches 1 16 64 256 --repeat 3 --output batch.json
python benchmarks/forward-derivatives/validate_batch_scaling.py --device cuda:1 --output batch-validation.json
```

Run an installed matching dev4 wheel with Torch and NumPy, without a source
`PYTHONPATH`. The benchmark driver imports helpers from the adjacent
`benchmark_forward_derivatives.py`. Both driver and helper use public APIs.

- [batch-scaling-20261021.json](../verification/batch-scaling-20261021/batch-scaling-20261021.json)
- [summary.csv](../verification/batch-scaling-20261021/summary.csv)
- [batch-validation-20261021.json](../verification/batch-scaling-20261021/batch-validation-20261021.json)
- [batch-smoke-20261021.json](../verification/batch-scaling-20261021/batch-smoke-20261021.json)
- [documentation-examples-20261021.json](../verification/batch-scaling-20261021/documentation-examples-20261021.json)
