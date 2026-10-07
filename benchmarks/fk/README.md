# FK performance comparison

[Home](../../README.md) · [简体中文](README.zh-CN.md) · [All median timings](TIMINGS.md)

This report compares SeismicX GRTM with [Lupei Zhu's FK](https://github.com/rwalkerlewis/fk)
on Apple CPU, Apple Metal/MPS, remote CPU and one NVIDIA RTX 5090. It includes
single-thread and multithread FK baselines, workload-dependent speedups and the
individual measurements behind the figure.

**Version scope:** these measurements use the frozen SeismicX GRTM **0.5.0**
scientific snapshot `e7283a90092652f2037e779540d795edd3d404c5` and FK commit
[`cde4086f8cb919983c4bcc88fb63b84c8ba24cb1`](https://github.com/rwalkerlewis/fk/tree/cde4086f8cb919983c4bcc88fb63b84c8ba24cb1).
They are not a new performance test of the **0.5.1** binary release. Installed-
package checks for that release are [documented separately](../../verification/README.md).

![Complete-call timings and matched speedups](../figures/performance-summary.png)

[Download publication-resolution PDF](../figures/performance-summary.pdf) ·
[Plotted values](data/figure-values.csv) · [Timing CSV](data/timings.csv)

## What the comparison measures

All implementations evaluate the same single-force problem and the same five
nonzero displacement basis channels, on the same frozen wavenumber grid within
each case. FK propagation, source and Bessel kernels are called through an FP64
integer-grid, in-memory adapter. FK already evaluates its propagation kernel
outside the receiver loop. The timings are **not** elapsed time of stock
`fk.pl`, nor do they include writing SAC files.

Upstream FK at the pinned commit is serial. The **FK OpenMP** results use an
additional adaptation developed for this comparison: independent frequencies
run in parallel, layer/model COMMON blocks are thread-private, and each worker
receives the model. Physical arithmetic is unchanged. This is an adapted
comparator, not a claim that stock FK has an OpenMP option. Reference source,
adapter source and FK binaries are not distributed in this repository.

| Host / device | Single-worker baselines | Parallel configurations |
|---|---|---|
| Apple M4 Max, 16 CPU cores | FK 1; GRTM C 1 | FK 16; GRTM C 16; Metal with 16 host workers |
| AMD Ryzen Threadripper PRO 3955WX, 16 cores / 32 logical CPUs, WSL | FK 1; GRTM C 1 | FK 32; GRTM C 32 |
| One RTX 5090 on the remote host | Same remote baselines | CUDA with 32 host workers |

One native worker means **one thread, without core pinning**. Remote runs were
performed under shared ambient load; other users' processes were not stopped.
Case-boundary telemetry cannot remove per-trial contention. Treat these as
observed shared-host timings, not isolated hardware peak measurements.

C, CUDA and FK use FP64; FK compilation uses `-O3 -fdefault-real-8
-fdefault-double-8`. Metal uses FP64 host propagation and compensated float-pair
GPU contractions. No fast-math or native-target CPU tuning is used. See the
[full registered protocol and model arrays](data/protocol.json).

The timer covers input preparation, integration, GPU transfers/synchronization,
host inverse FFT/damping, and returned signal/spectrum arrays. FK uses ctypes
and NumPy synthesis without disk output; GRTM includes its temporary model-file
and fixed-array overhead. Imports, solver/context setup, choosing the frozen
grid, validation and archive writes are excluded. Native sub-timings are
retained in the data but do not replace complete-call wall time.

Each of 15 workloads has 10 configurations: **two warmups and five timed calls**
per configuration. Configurations rotate between timed rounds; all trials,
including slow trials, are retained. This gives 750 timed calls and 300 warmups.

## Workloads and figure panels

| Panels | Variable | Fixed settings / interpretation |
|---|---|---|
| a–d | Receivers: 1, 16, 64, 256 | Four layers including half-space; source depth 8 km; 256 samples at 0.5 s |
| e | Layers: 4, 8, 16, 32, 64 | 64 receivers; same source, sampling and shared integration grid as a–d |
| f | Source depth: 1, 8, 25, 45 km | Four layers; 64 receivers near 10 km; 256 samples at 2 s |
| g | Nominal range: 1, 10, 100, 1000 km | Four layers; depth 8 km; 64 receivers; same 512 s window as f |
| h | Within-GRTM receiver reuse ablation | Separate M4 Max / 8-thread experiment; not an FK comparison |

Receiver/layer sweeps span distances 10–30 km; the one-receiver case is **10 km**.
The `nominal_range_km=20` summary label does not override that actual distance.
These sweeps use one common period of 2394.021400914341 km and 252506 total
wavenumber nodes. Layer profiles have nonzero material gradients; adding layers
does not merely split an identical medium. Qp and Qs are 10000 throughout.

Geometry sweeps span 0.9–1.1 times the nominal range. Each geometry case has its
own frozen grid, shared across implementations. Thus panels f/g reflect both
geometry and its selected quadrature workload. The common depth-8 km/range-10 km
case is measured once. The fixed 512 s geometry window is a computational
workload choice, not an assertion that every distant arrival/coda fits inside it.

Panels a/b divide same-host serial/parallel FK median time by device median
time. Panel c shows complete-call times. Panel d uses each host's GRTM C1
baseline for GRTM engines and FK1 for FK OpenMP. Panels e–g use parallel FK.
Ratios **above 1 mean GRTM is faster**; ratios below 1 mean it is slower.
These are ratios of medians. Whiskers use `baseline_min / device_max` through
`baseline_max / device_min`; they are observed trial ranges, not confidence
intervals. Cross-host ratios are not described as algorithmic speedups.

Panel h changes the receiver reuse switch on a fixed GRTM problem, with
1/4/16/32 receivers, 128 samples at 0.4 s, depth 8 km and a common integration
grid for the 10–100 km distance family. It reports medians of three calls for
forward, objective + VJP, and one L-BFGS-B iteration (two objective/gradient
evaluations). Target and operator construction are excluded. This controlled
ablation addresses reuse specifically; the maximum cross-implementation GPU
ratio also includes implementation, parallelism and hardware effects.

## Selected results

| Case / engine | Time (s) | Serial FK / engine | Parallel FK / engine | GRTM C1 / engine |
|---|---:|---:|---:|---:|
| 4 layers, 256 receivers — Apple Metal | 0.0808 | 42.27× | 3.58× | 35.87× |
| 4 layers, 256 receivers — RTX 5090 | 0.7387 | 43.54× | 2.85× | 23.88× |
| 64 layers, 64 receivers — Apple Metal | 0.4339 | See full table | 0.45× | See full table |
| 64 layers, 64 receivers — RTX 5090 | 1.3102 | See full table | 0.94× | See full table |

Parallel FK is faster in the two 64-layer GPU cases shown here. FK's own
OpenMP speedup over serial ranges from 10.56–12.06× on Apple and 7.26–15.27× on
the remote host across these workloads. No engine is uniformly fastest.
Use the [complete timing table](TIMINGS.md), including both CPUs, rather than
extrapolating a single maximum ratio to a different model.

## Numerical checks and limits

All five nonzero channels, full signals and full spectra were compared.
The maximum per-channel relative L2 discrepancies against same-host FK are
**5.531 × 10⁻⁵** for waveforms and **1.814 × 10⁻⁵** for spectra, below the declared
10⁻⁴ diagnostic threshold. All 120 serial/OpenMP FK comparisons are bitwise
identical; timed output hashes are stable. No failed consistency case was
removed. Details are in [validation.json](data/validation.json).

These are **same-grid agreement checks**, not proof of wavenumber-integral
convergence, accuracy at all periods, or a physical validation of every finite-Q
constitutive convention. See [numerical notes](../../docs/NUMERICAL_NOTES.md).

## Inspectable records

- [environment.json](data/environment.json): recorded host versions and case-boundary load/GPU snapshots.
- [protocol.json](data/protocol.json): full models, grids, configuration and timing definitions.
- [timings.csv](data/timings.csv): 150 configuration medians, extrema and same-host ratios.
- [trials.csv](data/trials.csv): all 750 timed calls and output hashes.
- [warmups.csv](data/warmups.csv): all 300 warmup calls.
- [figure-values.csv](data/figure-values.csv): all 144 values in the combined figure.
- [reuse-ablation.json](data/reuse-ablation.json): panel h trial times and update diagnostics.
- [Data dictionary](data/README.md): units, grouping and field definitions.

The records permit independent recalculation of the published timing summaries.
They do not include the private benchmark adapters or the frozen solver
implementation needed to rerun this exact historical campaign. Running today's
wheel on a supplied model is a new experiment and should be labeled with its
own version, numerical settings, hardware and timing scope.
