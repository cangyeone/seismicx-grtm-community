# Differentiation and quadrature preview — 0.6.0.dev1

English | [简体中文](RESEARCH_PREVIEW.zh-CN.md)

The **0.6.0.dev1 GitHub prerelease** adds directional structural derivatives,
a native reverse VJP and independent wavenumber quadrature. Stable PyPI **0.5.1**
does not include these APIs. This is a research preview, not a new accuracy
certificate for every Earth model. Solver implementation remains binary-only.

## Installation and version selection

Download the wheel for your CPython version and operating system from the
[v0.6.0.dev1 prerelease](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev1),
then install its actual filename:

```sh
python -m pip install ./seismicx_grtm-0.6.0.dev1-<matching-tags>.whl
python -m pip install scipy python-flint
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

`<matching-tags>` is a placeholder, not a literal filename. Optional framework
adapters require `torch` or `jax` separately. This preview is distributed through
GitHub Releases; `pip install seismicx-grtm` continues to select the PyPI release.
See the release asset list and checksums for supported CPython/platform pairs.
Mac wheels include CPU + Metal; Linux/WSL x86_64 wheels include CPU + CUDA.
Structural derivatives and research quadrature run on CPU in both packages.
Existing source, batch, physical-field and receiver-function interfaces remain
available; see the [forward guide](PYTHON_FORWARD.md).

## Fixed-grid waveform derivatives

```python
import numpy as np
import grtm

model = grtm.example_model(units="si")
model.update(log2_samples=5, dt=0.4, distances=[10000., 30000.],
             t0=[0., 0.], ptam=0)
layers = np.asarray(model["layers"], dtype=float)
moment = grtm.strike_dip_rake(20, 40, 60, scalar_moment=1e15)
solver = grtm.Solver(backend="cpu")
probe = solver.operator(model, parameters=["vp", "vs"], threads=2)
grid = probe.freeze_grid(layers)
op = solver.operator(model, field="displacement", parameters=["vp", "vs"],
                     grid=grid, vjp_method="adjoint", threads=2)
linear = op.linearize(layers, moment)
waveform = linear.value
# Direction: +10 m/s in Vs in every original layer.
direction = np.zeros_like(layers)
direction[:, 2] = 10.
waveform_tangent = linear.jvp(direction)
# Gradient of the illustrative objective 0.5 * sum(waveform**2).
layer_gradient, moment_gradient = linear.vjp(waveform)
print(waveform.shape, waveform_tangent.shape, layer_gradient.shape)
```

Use the derivative of your actual weighted loss as the VJP cotangent. NED/SI
and impulse/source-history conventions follow the forward API. Layer directions
use model units. Selected `parameters` restrict the derivative: other columns
are zero by selection, not evidence of physical insensitivity.

| API / mode | Work and storage | Contract |
|---|---|---|
| `op.jvp(layers, moment, direction, moment_direction=None)` | One directional native traversal | Analytic chain rules; no model finite differences |
| `op.jacobian(layers, moment)` | Tangent columns; dense output | Use for individual sensitivity waveforms |
| `vjp_method="tangent"` | Contracts successive tangent columns | Lower derivative storage, one traversal per parameter |
| `vjp_method="adjoint"` | Reverse pass with a tape local to one node | All selected structural gradients; PTAM must be 0 |
| `vjp_method="auto"` | Adjoint for at least 8 selected parameters when supported and PTAM=0 | A heuristic, not a speed guarantee |
| `linearize()` | Saves the base value, grid and input snapshots | Propagation is recomputed for derivative products |

The first layer top is fixed. Keep layer count, source/receiver layer assignments
and numerical grid fixed within a differentiation stage. A constructor-level
`grid=` defines one discrete forward map throughout optimization. Without it,
the derivative freezes the locally selected grid and does not differentiate grid
selection. Refresh grids between stages only after checking convergence.
Source/interface crossings and changes in topology are excluded. Native reverse
VJP rejects PTAM; `auto` can retain the tangent path for PTAM. An explicit finite-
difference mode remains available as a validation/fallback option. Only first
structural derivatives are supported.

## PyTorch and JAX

`grtm.torch.Forward(model, parameters=["vp", "vs"], grid=grid,
vjp_method="adjoint")` uses the native reverse VJP for first-order backward.
The derivative is on CPU even with GPU forward or GPU tensors. Higher derivatives,
`torch.compile` and `torch.func/vmap` are unsupported.

```python
import jax
import jax.numpy as jnp
from grtm.jax import make_forward

jax.config.update("jax_enable_x64", True)
forward = make_forward(model, parameters=["vp", "vs"], grid=grid,
                       derivative_mode="vjp", vjp_method="adjoint", threads=2)
gradient = jax.jit(jax.grad(
    lambda earth: jnp.sum(forward(earth, jnp.asarray(moment)) ** 2)
))(jnp.asarray(layers))
```

JAX `derivative_mode="vjp"` avoids materializing the structural Jacobian and
supports first-order reverse `grad`, `jit` and sequential-callback `vmap`.
It recomputes the base native field in backward. It does **not** support JAX
forward-mode `jvp`/`jacfwd` or higher derivatives. The default
`derivative_mode="jacobian"` retains the dense-Jacobian custom-JVP path.
Batch interfaces do not fuse many models into one GPU kernel.

## Wavenumber integration

The native sampler uses **km, km/s and g/cm³** input units. Its outputs are raw
single-frequency channels before source radiation, inverse FFT or physical-field
synthesis. All methods below are CPU methods.

```python
native_model = grtm.example_model()
native_model.update(log2_samples=4, dt=0.4, distances=[10., 30.],
                    t0=[0., 0.], ptam=0)
with solver.wavenumber_kernel(native_model, frequency_index=4) as kernel:
    samples = kernel([0.03, 0.1, 0.3])  # [node, receiver, 30]
    result = kernel.integrate(2., method="gauss", atol=1e-9, rtol=0)
    assert result["certified"] is False
    print(result["value"].shape)       # [receiver, 30]
```

| Method | Numerical control | Guarantee scope |
|---|---|---|
| `gauss` | Adaptive paired Gauss rules | Finite-interval error estimate |
| `levin` | Bessel–Levin collocation; Gauss near zero | Sampled residual estimate |
| `shanks` | Batched uniform panels, compensated sums, guarded iterated three-point transforms | Finite-budget extrapolation diagnostic |

Gauss/Levin budget exhaustion raises an error. `kernel.coefficients(k)` returns
`[node, receiver, 30, 2]` Bessel coefficients and requires strictly positive k;
use direct samples near zero where coefficients can cancel. Shanks uses `cutoff`
and `panel_width` as the work budget; `atol`/`rtol` are not stopping tolerances.
`panels_per_batch=32` batches adjacent panels. `sequence_levels` reports accepted
transform levels; unsafe denominators fall back to lower levels or finite sums.
`transformed_input_error_estimate` propagates estimates, not validated bounds.

`solver.forward_quadrature(model, cutoff=2., method="gauss", sdr=(20,40,60),
scalar_moment=1e15, fields=("displacement",), atol=1e-8)` performs physical
synthesis with the usual high-level SI defaults. **Cutoff is always in km⁻¹.**
A full-waveform cutoff may also be supplied per sampled nonnegative frequency.
`atol` controls raw spectral channels, not metres or pascals. The native Levin
wrapper allocates absolute tolerance to its two subintervals; `rtol` does not
relax that wrapper's budget.

Native integration results remain `certified=False`; their unbounded tails and
kernel rounding are not certified. `require_certified=True` raises. Equal-depth
direct terms can require Abel interpretation and analytic subtraction; a dedicated
subtraction solver is not implemented. Arbitrary finite-Q inputs can violate the
material assumptions needed for finite-wavenumber regularity. There is **no
PyTorch/JAX backward adapter for adaptive quadrature**: the products above belong
to the separate fixed-grid operator.

## Independent references and limited certificates

```python
from grtm.quadrature import certified_hankel_reference, psv_high_k_bounds

reference = certified_hankel_reference("1", "2", power=1, atol="1e-12")
assert reference["certified"]
certificate = psv_high_k_bounds(native_model, 4, relative_radius=1e-4)
assert certificate["certified"]
```

The first certificate encloses only the analytic family
`k**power * exp(-a*k) * J_order(r*k)`, with a>0 and nonnegative integer power/order.
The second verifies high-wavenumber normalized P–SV matrices and grouped
recursion inverses over an admissible material box, with depths fixed. It does
not certify sampled native values, finite-k poles, or a full Green-function
integral. Invalid material classes are rejected. These flags have different
scopes from a native integration's `certified` flag.

Report version, model, units, grid/cutoff, backend and tolerances with results.
Extreme-long-period gradients and arbitrary-model accuracy remain subject to
convergence checks. The existing FK timings describe the frozen older benchmark
snapshot and are not measurements of this preview.
