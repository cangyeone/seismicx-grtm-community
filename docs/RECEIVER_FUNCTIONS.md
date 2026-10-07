# Receiver functions: H--kappa to waveform inversion

English | [简体中文](RECEIVER_FUNCTIONS.zh-CN.md)

Install inversion dependencies with `python -m pip install 'seismicx-grtm[rf]==0.5.1'`. NumPy handles synthesis and H–κ stacking; SciPy is loaded for inversion.

## A complete reproducible workflow

```python
import numpy as np
import grtm

truth = {"layers": [[0, 6300, 6300/1.782, 2800],
                    [34450, 8000, 4500, 3300]]}
p = np.linspace(0.04, 0.08, 12) / 1000  # s/m
observed = grtm.synthetic_rf(truth, p, dt=0.05, n_samples=768, t0=-5)
result = grtm.receiver_function_workflow(
    observed,
    h=np.arange(28000, 42001, 1000),
    kappa=np.arange(1.6, 1.901, 0.02),
    vp=6300,
    hk_options={"bootstrap": 100, "seed": 7},
    inversion_options={"fit_window": (1, 26)},
)
print(result["hk"]["H"], result["hk"]["kappa"])
print(result["model"]["layers"])
print(result["fit"]["success"], result["fit"]["rms"])
```

The workflow searches H--kappa, builds a crust-over-mantle starting model, and
fits the RF waveforms. By default it fits **crustal thickness and Vp/Vs** while
holding average crustal Vp, density, and mantle properties fixed. This is a
controlled two-parameter inversion, not independent recovery of every material
property. The H--kappa grid endpoints become optimization bounds.

The complete example above uses synthetic data. Save arrays with `numpy.savez`;
synthetic recovery alone does not establish field-data validity.

## Model, waveform, and unit conventions

Rows are `[top_depth, Vp, Vs, density]` or the existing six-column form with
`Qp, Qs`. Tops begin at zero and increase; the last layer is a half-space.
The solver is **elastic**: optional Q columns are ignored explicitly and output
metadata reports `attenuation="elastic"`. Positive shear and bulk moduli are
required. Only horizontal, isotropic solid layers, a traction-free surface,
and subcritical bottom-incident P waves are supported. Fluid/ocean layers,
dip, anisotropy, attenuation, and S receiver functions are outside this solver.

Default `units="si"`: depths m, velocities m/s, density kg/m³, horizontal
slowness **s/m**. `units="native"` uses km, km/s, g/cm³ and s/km. Time and
Gaussian settings are identical in either system. Slowness in s/degree must
be converted separately; this API does not compute spherical-Earth arrivals.
The ray direction must satisfy p*Vp < 1 in every layer.

R points along horizontal propagation, **away from the source**; Z points up.
This RF convention is explicit and differs from the point-source NED down
axis. Rotate N/E observations using the station-to-event backazimuth beta:
`R = -N*cos(beta) - E*sin(beta)` and `Z = -D`, with beta in radians.
The output is a dimensionless radial RF `[ray, sample]`, with time relative
to direct P. `t0` is the first output time; the window must include zero.
There is no automatic P picking, rotation, instrument removal, or data download.

Plane-wave synthesis uses a stable 2x2 scattering recursion, including
interface conversions and free-surface multiples, batched over rays and
frequencies. It does not perform the point-source wavenumber integral. This
new solver runs in **NumPy on CPU**, regardless of bundled C/CUDA/Metal engines;
the existing point-source `Solver` and framework adapters are unchanged.

RFs use water-level R/Z spectral deconvolution and the Gaussian
`exp[-omega²/(4*a²)]`; `gaussian=a` is in rad/s, not Hz. The filtered vertical
autodeconvolution has unit amplitude at lag zero and sets the RF normalization.
`radial_spectrum` and `vertical_spectrum` in synthesis are unfiltered surface
displacement transfer responses to unit half-space P incidence; `spectrum` is
the filtered/normalized RF before the output time shift.

FFT padding defaults to at least four record lengths. This reduces circular
wraparound but is not an infinite-time solution: check `pad_factor=8` or longer
records for long coda or highly reflective models. Match Gaussian bandwidth,
normalization and signs between observations and synthetics. Water-level
regularization depends on reference spectral power; band-limited sources and
noise can leave differences from ideal impulse synthetics.

## Observed records and H--kappa

```python
# radial and vertical: preprocessed/P-windowed [event, sample] arrays
observed = grtm.receiver_function(
    radial, vertical, dt=0.05, t0=-5,
    slowness=p, gaussian=2.5, water_level=0.01,
)
hk = grtm.hk_stack(observed, h=np.arange(20000, 60001, 250),
                   kappa=np.arange(1.5, 2.001, 0.005), vp=6300)
initial = grtm.hk_initial_model(hk)
fit = grtm.invert_receiver_functions(observed, initial, fit_window=(1, 26))
```

The input waveforms must already share component signs and response units;
remove instrument response, detrend, taper, and select the P window before
deconvolution. `t0` controls the deconvolution output window; a common input
arrival delay cancels in R/Z. For precomputed RF arrays supply `slowness`,
`time` (uniform seconds relative to P), and matching filter/normalization
settings yourself. Metadata from `receiver_function` or `synthetic_rf` is
inferred automatically; inconsistent overrides are rejected by inversion.

Let qP=sqrt(1/Vp²-p²) and qS=sqrt(kappa²/Vp²-p²). The three predicted delays
are H(qS-qP), H(qS+qP), and 2HqS. The stack is
`w1*RF(tPs) + w2*RF(tPpPs) - w3*RF(tPpSs+PsPs)`, averaged over records.
Phase weights default to (0.7, 0.2, 0.1). Grid cells lacking any phase in any
record are invalid and returned as NaN; extend the window if none are valid.
Outputs include the surface, validity mask, phase times, boundary-maximum
flag, and optional record bootstrap estimates. Bootstrap intervals condition
on assumed Vp, weighting, and phase identification; they do not include model
error. A boundary maximum calls for examining the grid and data.

## Multilayer refinement, priors, and diagnostics

```python
fit = grtm.invert_receiver_functions(
    observed, initial_multilayer_model,
    parameters=((0, "thickness"), (1, "thickness"), (1, "kappa")),
    bounds={(0, "thickness"): (500, 10000),
            (1, "thickness"): (15000, 45000), (1, "kappa"): (1.5, 2.0)},
    prior_std=(2000, 8000, 0.15), fit_window=(0.5, 26),
)
```

Use direct inversion for a separately constructed multilayer initial model;
H--kappa does not identify intracrustal layering. Parameters are
`(zero_based_layer_index, "thickness"|"vp"|"kappa"|"density")`. Thickness
exists only for finite layers; changing it moves all deeper tops. Vp and
kappa are independent coordinates, with Vs=Vp/kappa. Q and layer count are
fixed. All bounds must contain the initial values; defaults are 0.5--1.5 times
the initial value and 1.4--2.2 for kappa. Vp bounds permitting critical
incidence are rejected before optimization. `prior_std` applies Gaussian
priors about initial fitted values, in the same parameter units.

The optimizer uses SciPy `least_squares`, a **central finite-difference
Jacobian** (one-sided near bounds), parameter scaling and optional robust
loss. This RF Jacobian is separate from the native point-source AD engine.
`noise_std` supplies positive per-record amplitude scales; `event_weights`
supplies relative record weights. No higher-order/PyTorch/JAX RF gradient
interface is provided in this version.

Inspect `success`, `message`, `initial_rms`, `rms`, `active_bounds`,
`data_rank`, and `scaled_singular_values`. `success` means numerical optimizer
termination, not uniqueness. Conditional local covariance is returned only
for a full-rank, unregularized linear-loss fit; it is scaled by residual
variance and excludes correlations in RF noise, assumed Vp/mantle errors,
and model misspecification. Priors may stabilize a fit without making its
data independently informative. Try multiple starting models and inspect
waveform residuals before interpreting field data.

## Method references and verification

- [Zhu & Kanamori (2000), H--kappa method](https://doi.org/10.1029/1999JB900322).
- [Receiver functions and plane-wave component responses](https://doi.org/10.1093/gji/ggz002).
- [Reappraisal of H--kappa assumptions and ambiguity](https://academic.oup.com/gji/article/219/3/1491/5543902).

The implementation is newly written from elastic stress/displacement boundary
conditions, without copying third-party receiver-function code. Tests compare
scattering spectra to an independent elastic ODE/matrix-exponential boundary
solver; check homogeneous media, transparent layer splitting, Ps/multiple
arrival times and polarities, units, FFT padding, source cancellation, H--kappa
on independently generated pulses, and off-grid two-layer/multilayer recovery.
