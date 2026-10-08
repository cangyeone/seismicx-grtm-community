# Public API reference — stable 0.5.1 and preview additions

[Home](../README.md) · [Worked examples](QUICKSTART.md) · [Forward guide](PYTHON_FORWARD.md)

Install `seismicx-grtm`; import `grtm`. This is a compact reference to the binary
release's public interfaces. NumPy is required. Torch, JAX and SciPy are optional
and are loaded for their respective adapters or inversion functions.

This page's original signatures describe stable **0.5.1**. Preview **0.6.0.dev1**
adds `jvp()`, `linearize()`, constructor `grid=`, `vjp_method=`, JAX
`derivative_mode="vjp"`, `wavenumber_kernel()`, `compute_quadrature()` and
`forward_quadrature()`. See the [versioned preview guide](RESEARCH_PREVIEW.md)
for complete examples, shape/unit conventions and restrictions. These new
interfaces require an explicit preview wheel installation.

## Models and engine selection

| Entry point | Purpose |
|---|---|
| `grtm.example_model(units="native")` | Return an example model dictionary; request `units="si"` for high-level SI calculations |
| `grtm.Solver(backend="cpu")` | Create a solver; aliases `c`, `cuda`, `metal`, `mps` select the corresponding engine |
| `grtm.write_input(config, path)` | Write a legacy model file in native units |
| `grtm.strike_dip_rake(strike, dip, rake, scalar_moment=1.0)` | Return the six NED moment components from angles in degrees |

`Solver(library=...)` is for a separately supplied, compatible native library.
Normal wheel users do not need it. Engine selection does not select a Torch/JAX
tensor device; configure that separately.

| Model key | Meaning |
|---|---|
| `layers` | Rows `[top_depth, vp, vs, density, Qp, Qs]`; first top is zero, final row is a half-space |
| `source_depth`, `receiver_depth` | Source and receiver depths, positive down |
| `distances` | Positive horizontal distances; one value per receiver |
| `dt` | Sampling interval in seconds |
| `log2_samples` | Integer base-two exponent of the time-series length |
| `t0` | Window start in seconds for each receiver |
| `taper`, `ptam` | Numerical controls; retain the example's defaults until familiar with the solver |

High-level SI uses metres, m/s and kg/m³. Native units are km, km/s and g/cm³.
`compute()`, the CLI and legacy files use native units. Time is always seconds.
The model's units must agree with the `units` argument; arrays carry no unit tags.

## Physical forward calculation

```python
result = solver.forward(
    model, moment_tensor=None, sdr=None, scalar_moment=1.0, force=None,
    azimuth=0.0, units="si", coordinates="ned",
    fields=("displacement", "strain", "stress"), stf=None,
    constitutive="kernel", grid=None,
    # Optional execution controls: threads, device, batch_size, ...
)
```

Supply exactly one source: `moment_tensor`, `sdr`, or `force`. Moment tensors are
six components `(Mnn, Mee, Mdd, Mne, Mnd, Med)` or symmetric 3×3 arrays; SI units
are N m. `sdr` contains strike, dip and rake in degrees; `scalar_moment` supplies
its amplitude. Force is a three-component NED vector in N. `azimuth` is clockwise
from north, either scalar or one value per distance. Coordinates are `ned` or
`rtd` (radial, transverse, down).

| Result entry | Shape / meaning |
|---|---|
| `displacement` | `[distance, sample, 3]`, when requested |
| `gradient`, `strain`, `stress` | `[distance, sample, 3, 3]`, when requested |
| `spectrum` | Mapping from each requested field to its complex spectrum |
| `field_units`, `response` | Units and impulse/history response convention |
| `dt`, `t0`, `distances` | Sampling and receiver coordinates |

`gradient[..., i, j]` is ∂uᵢ/∂xⱼ. Stress is positive in tension. Without `stf`, SI
outputs are impulse kernels in m/s, 1/s and Pa/s. `stf` is a dimensionless source
**history**, sampled at `dt`, not a moment rate; convolution multiplies by `dt`
and requires zero `t0`. History outputs are in m, dimensionless strain and Pa.
See [constitutive and attenuation conventions](NUMERICAL_NOTES.md).

## Batches and raw basis functions

| Method | Return / default |
|---|---|
| `solver.forward_batch(models, workers=1, stack=True, **forward_options)` | Stacked physical fields, or a list with `stack=False` |
| `solver.iter_forward_batch(models, workers=1, **forward_options)` | Ordered iterator; accepts a generator and bounds in-flight work |
| `solver.compute(config_or_path, grid=None, tangent=None, **options)` | Legacy basis arrays in native conventions |
| `solver.compute_batch(models, workers=1, stack=False, **options)` | List of raw results by default |
| `solver.iter_batch(models, workers=1, **options)` | Ordered iterator of raw results |

`compute()` returns `signal` with shape `[distance, sample, 30]`, `spectrum`,
and calculation metadata. Raw channels are source basis functions, not final
NED seismograms. Prefer `forward()` for named physical outputs.
Batch calls dispatch models individually; compatible output shapes are required
for stacking. `BatchError` identifies the failing model's index.

Common native controls include `threads` (zero means automatic), `device`
(zero-based GPU index), `batch_size` (GPU work chunk, default 512),
`period_length` (positive native-unit integration-period override; zero means
automatic) and `max_ptam_steps` (default 100000). These are numerical/execution
settings, not physical model parameters. Thread counts apply to host work even
with a GPU engine. A `grid` is an opaque frozen-grid object from the operator,
not an arbitrary user-defined list of quadrature points.

## Structural derivatives and forward operators

```python
op = solver.operator(
    model, field="displacement", domain="signal", azimuth=0.0,
    units="si", coordinates="ned", stf=None, constitutive="kernel",
    parameters=("depth", "vp", "vs", "rho", "qp", "qs"),
    method="analytic", relative_step=1e-3, absolute_step=0.0,
)
y = op(layers, moment_tensor)
grid = op.freeze_grid(layers)
jac = op.jacobian(layers, moment_tensor, grid=grid, check_step=False)
layer_grad, moment_grad = op.vjp(layers, moment_tensor, cotangent, grid=grid)
```

`layers` has shape `[L, 6]`, or `[B, L, 6]` for a batch. Sources are shared `[6]`
or per-model `[B, 6]`. `domain` is `signal` or `spectrum`; `field` selects one
physical field. Parameters may be names covering all eligible layers or pairs
such as `[(0, "vp"), (0, "vs")]`. The first interface at zero is fixed.

`jac["layers"]` has `output_shape + (L, 6)` and `jac["moment_tensor"]` has
`output_shape + (6,)` for one model. Unselected columns are zero, which means
they were not differentiated. `check_step=True` adds independent central-
difference evaluations; it checks derivative consistency, not quadrature error.

The default analytic method uses CPU tangent propagation on a fixed local
integration grid. `method="finite_difference"` selects numerical differences.
VJP contracts parameter derivatives without retaining the full structural
Jacobian. Only first-order derivatives are supported. Fixed topology and
source/receiver layer membership are required; geometry, layer count and
sampling configuration are not differentiable arguments.

The following framework table describes stable 0.5.1; the preview adds optional
native adjoint VJP and JAX matrix-free reverse paths as described above.

| Interface | Forward | Structural backward | Derivative storage |
|---|---|---|---|
| NumPy `ForwardOperator.vjp` | Selected native engine | CPU tangent integrals | Streamed contraction |
| `grtm.torch.Forward` | Selected native engine with host transfers | CPU tangent VJP | Streamed contraction |
| `grtm.jax.make_forward` | Host callback to selected engine | CPU dense-Jacobian callback | O(output cells × L × 6) |

Torch provides first-order autograd. JAX provides `jit`, first-order `grad`/JVP/
`jacfwd` and sequential-callback `vmap`; it is not an XLA-native wave solver.
See the [framework examples](PYTHON_FORWARD.md#pytorch) and
[Apple MPS guide](APPLE_METAL.md).

## Receiver functions and H–κ inversion

This separate elastic plane-wave solver runs on NumPy/CPU. SI slowness is s/m,
not s/km or s/degree. It uses R away from the source and Z up; these differ
from NED's down axis. The model may have four or six columns; Q is ignored.

| Function | Main arguments / outputs |
|---|---|
| `synthetic_rf(model, slowness, ...)` | Synthesize radial receiver functions and component transfer spectra |
| `receiver_function(radial, vertical, ...)` | Water-level spectral R/Z deconvolution of preprocessed records |
| `hk_stack(data, h=..., kappa=..., vp=..., ...)` | H–κ surface, best H/κ, validity mask and optional record bootstrap |
| `hk_initial_model(hk, crust_density=None, mantle=None)` | Construct a crust-over-mantle initial model |
| `invert_receiver_functions(data, initial_model, ...)` | Fit selected structural parameters and return model/residual diagnostics |
| `receiver_function_workflow(data, h=..., kappa=..., vp=..., ...)` | H–κ → initial model → waveform inversion |

Synthesis defaults: `dt=0.05`, `n_samples=1024`, `t0=-5`, `gaussian=2.5`,
`water_level=0.01`, `pad_factor=4`, `units="si"`. Gaussian is in rad/s.
H–κ defaults: phase `weights=(0.7, 0.2, 0.1)`, `bootstrap=0`, `seed=0`.
Inversion defaults: `parameters=((0,"thickness"),(0,"kappa"))`,
`fit_window=(1,25)`, `loss="linear"`, `max_nfev=100`, `noise_std=1`.

Inversion supports thickness, vp, kappa and density; its Jacobian uses finite
differences, independently of the point-source tangent engine. Covariance is
a conditional local diagnostic, not a guarantee of geological resolution.
Full conventions, optional bounds/priors/weights and runnable workflows are
in the [receiver-function guide](RECEIVER_FUNCTIONS.md).
