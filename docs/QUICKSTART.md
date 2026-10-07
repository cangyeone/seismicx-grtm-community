# Quick start

[Home](../README.md) · [Installation](INSTALLATION.md) · [Full forward API](PYTHON_FORWARD.md)

These examples call the installed public API. They do not implement the solver.

## 1. Create a model and physical source

```python
import numpy as np
import grtm

model = grtm.example_model(units="si")
model.update(log2_samples=6, dt=0.2, distances=[10000., 30000.], t0=[0., 0.])
solver = grtm.Solver(backend="cpu")
mt = grtm.strike_dip_rake(20., 40., 60., scalar_moment=1e15)
result = solver.forward(
    model, moment_tensor=mt, azimuth=[0., 45.],
    fields=("displacement", "gradient", "strain", "stress"), threads=4,
)
print(result["displacement"].shape)  # (2, 64, 3)
print(result["stress"].shape)        # (2, 64, 3, 3)
print(result["field_units"])
```

There are `2**log2_samples` time samples. Here the record is short to keep the
example fast. Scientific calculations must cover the arrivals/coda of interest
and resolve the desired bandwidth. A layer row is
`[top_depth_m, Vp_m_s, Vs_m_s, density_kg_m3, Qp, Qs]`; the final row is a half-space.
`source_depth`, `receiver_depth`, and `distances` are in metres in this high-level
interface. `t0` has one value per receiver.

Pass exactly one of `moment_tensor`, `sdr` or `force`. Tensor components are
`(Mnn, Mee, Mdd, Mne, Mnd, Med)` in N m. A symmetric 3×3 tensor is also accepted.
Strike/dip/rake are degrees; azimuth is clockwise from north. Default output
coordinates are north, east, down. `coordinates="rtd"` returns radial,
transverse, down. Strain is tensor strain; off-diagonal values are not engineering
shear strain. Stress is positive in tension.

## 2. Apply a source history and save outputs

Without `stf`, outputs are impulse kernels in m/s, 1/s and Pa/s. A dimensionless
source amplitude history is convolved causally with the impulse kernel using dt:

```python
time = np.arange(1 << model["log2_samples"]) * model["dt"]
history = np.minimum(time / 0.5, 1.0)
response = solver.forward(
    model, moment_tensor=mt, stf=history, azimuth=45,
    fields=("displacement", "strain", "stress"), threads=4,
)
np.savez(
    "physical-response.npz", time=time, distances=model["distances"],
    displacement=response["displacement"], strain=response["strain"],
    stress=response["stress"],
)
with np.load("physical-response.npz") as saved:
    print(saved["displacement"].shape)
```

`stf` represents moment/force **history**, not moment rate. Integrate a moment-rate
history first. Source convolution requires all `t0=0`; nonzero window origins
would omit the beginning of the causal history. Save field units and model
metadata with your project so an array cannot be mistaken for a differently
scaled quantity.

## 3. Evaluate many Earth models

```python
from copy import deepcopy
models = []
for factor in (0.99, 1.0, 1.01):
    trial = deepcopy(model)
    trial["layers"][0][2] *= factor
    models.append(trial)
batch = solver.forward_batch(
    models, moment_tensor=mt, fields=("displacement",),
    workers=2, threads=2,
)
print(batch["displacement"].shape)  # (3, 2, 64, 3)
```

For a large generator, use `iter_forward_batch()` and consume one result at a
time. Results retain input order. Stacking requires compatible shapes; use
`stack=False` for heterogeneous records. Avoid multiplying a large worker count
by a large native-thread count. Models are dispatched separately; a batch is
not one fused GPU operation.

## 4. Get a structural Jacobian or a loss gradient

```python
layers = np.asarray(model["layers"], dtype=np.float64)
op = solver.operator(
    model, field="displacement", azimuth=45,
    parameters=[(0, "vp"), (0, "vs")], threads=4,
)
y = op(layers, mt)
jac = op.jacobian(layers, mt)
print(jac["layers"].shape)  # y.shape + (number_of_layers, 6)
cotangent = 2.0 * y / y.size
layer_gradient, source_gradient = op.vjp(layers, mt, cotangent)
```

This example differentiates mean squared amplitude as a simple demonstration.
For data fitting, form a residual against your observations with appropriate
weights and units. Choose only parameters needed by the inverse problem.
The selected point-source structural derivatives use CPU tangents on a fixed
local integration grid. A dense Jacobian can be large; VJP is preferable for
scalar losses. See the [forward guide](PYTHON_FORWARD.md#structural-jacobians-and-sensitivity)
for topology constraints, independent checks and the finite-difference option.

## 5. Use the command line

The CLI exposes the legacy basis-function interface in **native units**:
km, km/s, g/cm³ and seconds. It is distinct from high-level SI `forward()`.

```sh
grtm --info
grtm --write-example model.json
grtm --model model.json --backend cpu --threads 4 --output green.npz
```

```python
import numpy as np
with np.load("green.npz") as saved:
    print(saved["signal"].shape)
    print(saved["spectrum"].dtype)
```

Raw `signal` has shape `[distance, sample, 30]`; it stores source basis channels,
not a final three-component seismogram. Use `Solver.forward()` to combine a
physical source, rotate coordinates and obtain named physical fields.
Use `--backend mps` on Apple Silicon or `--backend cuda` on a supported NVIDIA
machine. The CLI writes NumPy NPZ files, not SAC or MiniSEED.

Continue with the [PyTorch/JAX interfaces](PYTHON_FORWARD.md#pytorch) or the
[complete receiver-function inversion example](RECEIVER_FUNCTIONS.md).
