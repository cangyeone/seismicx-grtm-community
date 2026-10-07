# Apple Metal and MPS

English | [简体中文](APPLE_METAL.zh-CN.md) | [Home](../README.md)

Install the Apple Silicon wheel with `python -m pip install seismicx-grtm`.
It contains CPU and a **precompiled Metal library**. No Xcode or shader compiler
is required by the installed package. Apple M4 Max is the tested device.

```python
import grtm
solver = grtm.Solver(backend="mps")  # "metal" and "apple" are aliases
model = grtm.example_model(units="si")
model.update(log2_samples=6, dt=0.2, distances=[10000., 30000.], t0=[0., 0.])
result = solver.forward(model, sdr=(20, 40, 60), scalar_moment=1e15, threads=4)
print(solver.backend_stats())
```

This is a hybrid solver: layer propagation and precision-sensitive operations
run on the CPU, while receiver contractions use Metal. Multiple receivers may
amortize GPU overhead; small jobs can be faster on CPU. Use the
[matched FK comparison](../benchmarks/fk/README.md) to understand workload effects,
then time the complete API call for your own workload.

## Precision and memory

Host propagation uses FP64; GPU contractions use compensated float-pair
arithmetic, approximately 48 mantissa bits. This is not IEEE FP64 and has a
more limited exponent range. Quasistatic stabilization, optional PTAM, FFT and
structural tangent derivatives remain on the host. Extremely small normalized
amplitude components can be pruned; the statistics report the count. Overflow
and incompatible devices raise errors.

`backend_stats()` describes the calling thread's latest Metal solve. Its GPU
time excludes CPU work, so it is not the complete-call speed. `dispatches=0`
can be correct when all frequency bins use host stabilization.

`batch_size` controls Metal frequency batches rather than CUDA wavenumber tiles.
Large receiver sets are chunked. Output memory still grows with receiver count,
sample count and requested fields; use iterators for large model ensembles.

## PyTorch MPS tensors

```python
import torch
import grtm
from grtm.torch import Forward

model = grtm.example_model(units="si")
model.update(log2_samples=5, dt=0.2, distances=[10000.], t0=[0.])
earth = torch.tensor(model["layers"], device="mps", dtype=torch.float32,
                     requires_grad=True)
mt = torch.tensor(grtm.strike_dip_rake(20, 40, 60, 1e15), device="mps",
                  dtype=torch.float32, requires_grad=True)
forward = Forward(model, backend="mps", parameters=[(0, "vs")], threads=4)
y = forward(earth, mt)
y.square().mean().backward()
print(earth.grad.shape, mt.grad.shape)
```

Tensor `device="mps"` and solver `backend="mps"` must be chosen separately.
MPS tensor input/output uses float32 and complex64, while host physics retains
FP64. Host transfers and callbacks remain part of the operation. Structural
backward runs the CPU tangent VJP. Higher derivatives, `torch.compile` and
`torch.func/vmap` are unsupported.

`grtm.jax.make_forward(model, backend="metal", ...)` uses host callbacks and
can invoke Metal without a JAX Metal plugin. JAX's own arrays need not live on
the Apple GPU. JAX derivative callbacks construct dense Jacobians; they are
not the streamed PyTorch VJP implementation.

See [numerical notes](NUMERICAL_NOTES.md), [framework guide](PYTHON_FORWARD.md),
and [0.5.1 release validation](../verification/README.md).
