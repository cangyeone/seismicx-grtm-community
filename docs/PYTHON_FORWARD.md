# Python forward, batch, and sensitivity API

English | [简体中文](PYTHON_FORWARD.zh-CN.md)

Distribution: `seismicx-grtm`; import and CLI: `grtm`. See the [installation guide](INSTALLATION.md) for backend selection and migration from `grtm-green`.

## Installation and backends

```sh
python -m pip install seismicx-grtm==0.5.1
python -m pip install 'seismicx-grtm[torch,jax]==0.5.1'
```

The published wheel includes CPU + Metal on Apple Silicon, or CPU + CUDA on
Linux/WSL x86_64. No compiler is required. CUDA execution needs a compatible
NVIDIA driver and compute capability 12.0 (RTX 5090 tested). CPU is the default.
See [installation](INSTALLATION.md) and [Apple Metal](APPLE_METAL.md).
Importing `grtm` loads NumPy; optional Torch/JAX are imported by their adapters.

## Models, sources, coordinates, and units

High-level `forward()` and `ForwardOperator` default to `units="si"`:

| Input | SI | `units="native"` |
|---|---|---|
| Layer row | `[top_depth_m, Vp_m_s, Vs_m_s, density_kg_m3, Qp, Qs]` | `[km, km/s, km/s, g/cm³, Qp, Qs]` |
| Source/receiver depth, distances | m | km |
| dt, t0 | s | s |
| Moment tensor | N m | One unit = 10¹⁸ N m |
| Force | N | One unit = 10¹⁵ N |

`compute()`, `write_input()`, and legacy numerical files keep native units. `example_model()` returns a native model; `example_model(units="si")` returns an SI model. Physical forward and derivatives require the native library introduced in 0.3, including its moment-tensor corrections; old external libraries still support `compute()`.

Layer tops start at zero and increase strictly; the last layer is a half-space. Vp > Vs > 0, density and Q > 0. Fluid layers, zero distance, and differentiable changes in layer count are unsupported.

Coordinates are **NED**: north, east, down. Azimuth is clockwise from north; all angles are degrees. The six moment components are `(Mnn, Mee, Mdd, Mne, Mnd, Med)`; symmetric 3×3 tensors are also accepted. Positive rake denotes reverse slip. `strike_dip_rake()` generates a double couple; arbitrary symmetric moment tensors are supported. Supply exactly one of `moment_tensor`, `sdr`, or `force`.

```python
import grtm
model = grtm.example_model(units="si")
solver = grtm.Solver(backend="cpu")
mt = grtm.strike_dip_rake(20, 40, 60, scalar_moment=1e15)
result = solver.forward(
    model, moment_tensor=mt, azimuth=[0, 45, 90],
    fields=("displacement", "gradient", "strain", "stress"), threads=8,
)
# Alternatively: solver.forward(model, sdr=(20,40,60), scalar_moment=1e15, ...)
```

Displacement is `[distance, sample, 3]`; gradient/strain/stress are `[distance, sample, 3, 3]`. `result["spectrum"][field]` contains the corresponding complex spectrum. `dt`, `t0`, and `distances` describe sampling. `coordinates="rtd"` uses radial, transverse, down; default `"ned"` rotates both vectors and tensors.

`gradient[..., i, j] = ∂u_i/∂x_j`. Strain is the symmetric gradient and uses tensor shear strain, without the factor of two for engineering shear strain. Stress is positive in tension.

## Source histories and stress

Without `stf`, a source amplitude in N m or N gives an **impulse Green's kernel**: displacement m/s, strain 1/s, stress Pa/s, with `response="impulse_green"`. These kernels should not be labeled ordinary displacement or static stress.

`stf` is a dimensionless moment/force history h(t), sampled at dt from t=0. The physical history is M h(t) or F h(t), not normalized moment rate. Causal linear convolution multiplies by dt and retains the first n_samples points. Output then uses m, dimensionless strain, and Pa, with `response="history"`. Integrate a moment-rate history using dt before passing it. `stf=[1/dt]` reproduces impulse-kernel values. Convolution requires all t0=0; offset windows miss the causal beginning and are rejected.

```python
import numpy as np
n = 1 << model["log2_samples"]
t = np.arange(n)*model["dt"]
history = np.minimum(t/0.5, 1.0)
result = solver.forward(model, moment_tensor=mt, stf=history)
print(result["field_units"])  # m, 1, Pa
```

Default `constitutive="kernel"` matches the mathematical kernel: μ = ρ Vs0² is real; complex velocities a(ω), b(ω) follow the Q dispersion model; λ(ω) = μ[(a(ω)/b(ω))² − 2]; σ = λ tr(ε) I + 2με. The constitutive relation is applied at the kernel's complex frequencies, then transformed with the same damped FFT. `constitutive="elastic"` uses reference λ=ρ(Vp0²−2Vs0²), an explicit approximation at finite Q. Material parameters come from the receiver layer; a receiver exactly on an interface uses the upper side.

## Large Earth model batches

```python
for result in solver.iter_forward_batch(
    models, workers=4, threads=4, moment_tensor=mt,
    fields=("displacement", "stress"),
):
    save_one(result)

all_results = solver.forward_batch(models, moment_tensor=mt, workers=4, threads=4)
bases = solver.compute_batch(native_models, workers=4, stack=False)
```

Iterators accept generators, keep at most `workers` computations in flight, and preserve input order. `stack=True` gives `[model, distance, sample, ...]` and requires consistent shapes and metadata; use `stack=False` or iterators for heterogeneous models. `BatchError` includes the failed model index.

CPU default thread budgets are divided across workers; explicit `threads` is retained. GPU defaults to workers=1 and supports device selection. Batch calls dispatch each model separately; models are not fused into a single GPU kernel. Full-result memory grows with model count, so consume iterators for large jobs.

Tensor interfaces accept `[B,L,6]` layers and shared `[6]` or per-model `[B,6]` sources:

```python
op = solver.operator(model, field="displacement", azimuth=45)
y = op(earth_layers_batch, moment_tensor_batch)  # [B,D,N,3]
```

## Structural Jacobians and sensitivity

```python
layers = np.asarray(model["layers"], dtype=np.float64)
op = solver.operator(
    model, field="stress", domain="signal", azimuth=45,
    parameters=["vp", "vs", "rho", "qp", "qs", "depth"], threads=8,
)
j = op.jacobian(layers, mt, check_step=True)
# j["layers"]: output_shape + [L,6]
# j["moment_tensor"]: output_shape + [6]
# j["parameters"]: selected (layer_index, name) pairs
# j["relative_change"]: independent central-difference comparison
layer_gradient, source_gradient = op.vjp(layers, mt, cotangent)
```

Default **`method="analytic"`** uses first-order tangent code generated from the shared C kernel. The chain rule propagates through modal square roots, complex Q velocities, Lamé parameters, interface reflection/transmission, layer exponentials, sources, and integration. This is numerical wavenumber integration with analytic chain derivatives: differentiation does not perturb models or depend on a finite-difference step.

Each selected structural parameter runs one tangent integral. Linear moment-tensor derivatives reuse basis functions. Regular frequencies use double, ill-conditioned quasistatic frequencies use long double, and workspaces are reused within threads. Stress derivatives also include direct receiver-layer constitutive derivatives.

Optional `method="finite_difference"` uses two solves per parameter. Default `relative_step=1e-3`, `absolute_step=0`, with h=max(|p| relative_step, absolute_step). `check_step=True` adds an independent h/2 central-difference check, incurring additional solves. `relative_change` measures consistency, not physical integration error. Analytic `steps` are zero. Unselected parameters and the fixed first interface depth return zero, indicating they were not differentiated.

Both methods freeze the base integration period L and each frequency's integer wavenumber count to avoid discrete cutoff jumps:

```python
grid = op.freeze_grid(layers)
y_plus = op(layers_plus, mt, grid=grid)
y_minus = op(layers_minus, mt, grid=grid)
```

Layer count, source/receiver layer membership, and numerical branch topology remain fixed. Integer step counts and PTAM extremum selection are not differentiated. Interface-depth derivatives are rejected where an interface coincides with the source or receiver; differences crossing interfaces are also rejected. PTAM derivatives follow the base extremum/convergence branch and may be nonsmooth at switches.

Extreme long-period structural gradients remain unverified: see [numerical limits](NUMERICAL_NOTES.md). Extended precision alone does not establish reliability. Only first-order derivatives are supported. Jacobians are dense; VJP contracts each parameter without retaining the full structural Jacobian. Structural tangents execute on the **CPU**, including CUDA/Metal forward selection; source derivatives use NumPy linear synthesis. These are not GPU-native structural backward kernels.

## PyTorch

```python
import torch
from grtm.torch import Forward
f = Forward(model, backend="cpu", field="displacement", azimuth=45,
            parameters=["vp", "vs", "rho"], threads=8)
earth = torch.tensor(layers, dtype=torch.float64, requires_grad=True)
source = torch.tensor(mt, dtype=torch.float64, requires_grad=True)
y = f(earth, source)
loss = (y - observation).square().mean()
loss.backward()
# earth.grad, source.grad
# f.forward_sdr(earth, strike, dip, rake, M0) is also available.
```

`grtm.torch.strike_dip_rake()` preserves framework angle gradients. Batched layers, shared/per-model sources, and complex spectra are supported. Inputs need matching real floating dtype/device; output stays on that device. Use float64 on CPU/CUDA when practical.

`backend="cuda"` selects the native CUDA solver and involves host transfers. `backend="mps"` selects the hybrid Metal engine; MPS tensors use float32 and complex outputs use complex64. Tensor `device="mps"` and solver `backend="mps"` control different things and must be set separately. Structural backward uses the CPU tangent VJP and computes only selected `parameters`. Ordinary first-order autograd is supported; higher derivatives, `torch.compile`, and `torch.func/vmap` are not.

## JAX

```python
import jax
import jax.numpy as jnp
from grtm.jax import make_forward
jax.config.update("jax_enable_x64", True)
f = make_forward(model, backend="cpu", parameters=["vp", "vs", "rho"], threads=8)
y = jax.jit(f)(jnp.asarray(layers), jnp.asarray(mt))
grad_earth, grad_source = jax.grad(
    lambda earth, source: jnp.sum(f(earth, source)**2), argnums=(0,1)
)(jnp.asarray(layers), jnp.asarray(mt))
```

`pure_callback` + `custom_jvp` supports jit, first-order grad/JVP/jacfwd, sequential-callback vmap, and explicit Earth batches. `grtm.jax.strike_dip_rake()` preserves angle gradients. This is not an XLA-native solver and does not differentiate geometry, dt, STF configuration, or higher orders. The application controls JAX precision; the package does not change global settings.

JAX derivative callbacks construct dense Jacobians with O(output_cells × L × 6) memory, including unselected column positions. Prefer NumPy/PyTorch VJP for large outputs. vmap does not fuse models into one GPU kernel.

References: [PyTorch custom autograd](https://docs.pytorch.org/docs/stable/notes/extending.html), [JAX callbacks](https://docs.jax.dev/en/latest/notebooks/external_callbacks.html), [custom JVP](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html), and [Pyrocko NED moment tensors](https://pyrocko.org/docs/current/library/examples/moment_tensor.html).
