# Oscillatory wavenumber quadrature: 0.6.0.dev6

English | [简体中文](QUADRATURE_0.6.0.dev6.zh-CN.md)

Optimizes the existing Bessel–Levin scheme, adds joint field/analytic-JVP Levin integration and an optional `method="auto"`. See [Levin (1997)](https://www.tau.ac.il/~levin/OscIntAnal.pdf) for Bessel collocation and [Chen, Serkh and Bremer](https://arxiv.org/abs/2211.13400) for adaptive Chebyshev/truncated-SVD analysis. Their assumptions do not certify every native GRTM infinite integral.

Adjacent panels share kernel calls. Material-independent quadrature bases and real SVD factors are cached (256 geometry entries maximum); every call recomputes the material-dependent propagation. All complex channels and field/JVP pairs reuse those real factors. Independent residual samples, the `1e-13` SVD cutoff, roundoff diagnostic floors, worst-panel subdivision and exhaustion errors remain. No parameter finite differences or adaptive-mesh derivatives are introduced.

`auto` selects Levin when `cutoff * max(distance) > 512`, otherwise batched Gauss. This is a cost heuristic, not an accuracy shortcut or an assurance of fastest execution. Explicit selection is available; the default remains Gauss.

## CPU results

Apple M4 Max, baseline source `54cf35f`, identical native library/precision. Four-layer field cases and two-layer joint cases, all 30 raw moment-tensor coefficients. Finite interval `[0,4] km^-1`, atol=1e-10, rtol=0, order=12; five warmed repetitions, medians. Each repetition recomputes every physics sample. JSON retains first-call and all repeat timings, models, complex outputs and relative errors. Reference: a different Gauss mesh, order20, atol=1e-12. Table errors are absolute raw-coefficient differences. Equal-depth joint cases add the analytic Abel direct field; quasistatic cases use the existing stable modal representation.

| Case | Old method / ms | New method / ms | Ratio | Old / new nodes | Max field / JVP absolute difference vs reference |
|---|---:|---:|---:|---:|---:|
| near | levin 24.058 | Levin 7.801 | 3.08× | 1388 / 1388 | 2.52e-14 |
| far | levin 24.572 | Levin 7.767 | 3.16× | 1388 / 1388 | 6.04e-14 |
| shallow_far | levin 26.093 | Levin 8.138 | 3.21× | 1438 / 1438 | 4.31e-14 |
| joint_separated | gauss 188.974 | Levin 17.962 | 10.52× | 9360 / 1488 | 1.82e-13 / 3.67e-14 |
| joint_equal_depth | gauss 1879.250 | Levin 239.727 | 7.84× | 6408 / 1388 | 2.1e-13 / 3.32e-13 |
| joint_quasistatic | gauss 1143.052 | Levin 56.145 | 20.36× | 3888 / 610 | 1.06e-13 / 1.11e-13 |

Near receivers (10/30 km): new Gauss 2.020 ms versus new Levin 7.801 ms; auto selects Gauss. Far receivers (100/1000 km): 33.994 versus 7.767 ms (4.38×). Shallow/far: 101.378 versus 8.138 ms (12.46×). Old Gauss to new Levin combines an algorithm change and implementation improvements.

## Native PTAM, full waveform

Separate fixed model: four layers, source8 km, receivers10/30/100 km,16 samples,dt0.4 s,30 channels. Eight PTAM extrema, one native CPU thread. Wall time includes model setup and FFT; three warm repetitions. Gauss reference: order20,atol1e-13,cutoff8, with cutoff4-to8 changes retained. New Levin/auto: cutoff4,atol1e-12.

| Method | Period length (km) | ms | Max absolute spectrum error | Relative L2 spectrum error |
|---|---:|---:|---:|---:|
| levin | — | 158.118 | 2.45e-15 | 7.59e-12 |
| auto | — | 38.751 | 2.45e-15 | 7.51e-12 |
| ptam | 400.0 | 2.731 | 1.25e-07 | 0.000585 |
| ptam | 800.0 | 4.642 | 6.39e-08 | 0.000174 |
| ptam | 1600.0 | 7.819 | 6.16e-08 | 0.000151 |
| ptam | 3200.0 | 12.876 | 7.74e-08 | 0.000202 |
| ptam | 6400.0 | 24.260 | 1.36e-07 | 0.00025 |
| ptam | 12800.0 | 46.715 | 1.35e-07 | 0.000248 |

No blanket matched-error speedup over PTAM is established: coarse native PTAM is faster here, but its tested settings do not reach the new quadrature's errors, and step refinement does not eliminate its error plateau. Legacy PTAM and the independent sampler also use different Bessel implementations; this is an end-to-end comparison, not an isolated test of the PTAM transformation. Auto avoids the unnecessary Levin solve cost in this near-field workload.

Extra diagnostics are retained: a1000 km receiver with an excessively short time window has a nearly zero damped field, making relative errors uninformative; a long-window low-frequency native Gauss reference exhausted2048 panels at1e-13 and was rejected, excluded from speed claims. Stable low-frequency joint field/JVP verification remains separate.

## Usage and scope

```python
import grtm
import numpy as np

solver = grtm.Solver(backend="cpu")
model = grtm.example_model()  # native units: km, km/s, g/cm^3
model.update(log2_samples=5, dt=0.4, distances=[100., 1000.],
             t0=[0., 0.], source_type=1, ptam=0)
with solver.wavenumber_kernel(model, frequency_index=4) as kernel:
    field = kernel.integrate(4., method="auto", atol=1e-10, rtol=0)
    direction = np.zeros_like(model["layers"])
    direction[0, 2] = 1.  # top-layer Vs direction, analytic chain rule
    joint = kernel.integrate_jvp(4., direction, method="auto",
                                 atol=1e-10, rtol=0)
    print(field["method"], field["kernel_calls"], field["evaluations"])
    print(joint["value"].shape, joint["jvp"].shape)
```


Cutoff is always km^-1. Check cutoff refinement in production. Results are `[receiver,30]` native coefficients. Full raw waveforms: `solver.compute_quadrature(model, cutoff=4., method="auto", ...)`; NED/SI synthesis: `forward_quadrature`.

This adaptive integration path is CPU-only, including with a CUDA/Metal Solver library. Existing fixed-grid CUDA forward/derivative APIs are unchanged. Surface-wave CUDA performance is a [separate report](PERFORMANCE_0.6.0.dev6.md).

Error estimates cover the finite interval, without rigorous infinite-tail or total kernel-roundoff enclosures. `certified=False`, `tail_bound=None`, and rejection of `require_certified=True` remain. Raw equal-depth channels need not have an ordinary improper integral; joint integration retains analytic direct-field recovery.

Evidence: [CPU data](../verification/performance-0.6.0.dev6/quadrature/cpu.json), [PTAM data](../verification/performance-0.6.0.dev6/quadrature/ptam-cpu.json), [rejected low-frequency reference](../verification/performance-0.6.0.dev6/quadrature/ptam-low-frequency-failure.json). Private reproduction scripts: `measure.py`, `ptam_check.py` in the same directory.
