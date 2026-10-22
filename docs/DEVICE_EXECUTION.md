# Device execution and batch interfaces — 0.6.0.dev7

dev10 adds [fused CUDA Green batches and an explicit fast precision policy](CUDA_GREEN_BATCH.md). The table below retains the dev7 baseline; RF, dispersion and MPS boundaries are unchanged.

English | [简体中文](DEVICE_EXECUTION.zh-CN.md)

Select one compute device for both forward and structural derivatives. RF and
surface-wave CUDA paths keep their numerical propagation, roots, kernels and
backward work on CUDA. Green-function CUDA now also performs analytic structural
integrals on CUDA, with the specific precision exceptions below. GPU failures
are reported; CUDA is never silently replaced by a CPU solver.

| Family | CPU | CUDA | Apple / MPS |
|---|---|---|---|
| Point-source Green forward | Native FP64, low-frequency long double | FP64 GPU integrals; Torch CUDA FFT/synthesis on GPU; host precision exceptions | Existing hybrid: host FP64 propagation, compensated float-pair Metal receiver integration |
| Green structural derivatives | Native tangent / node-local reverse AD | Native analytic tangents and true reverse AD; low-frequency host exceptions | CPU structural AD |
| Joint RF forward + VJP | Native complex128 preparation, propagation and material adjoints; Torch FFT | Native complex128 preparation, propagation, material adjoints and Torch FFT on CUDA | Explicit CPU FP64 bridge; real output MPS float32, complex spectra remain CPU complex128 |
| Surface phase/group + kernels | Native FP64 characteristic adjoints and dual/hyper-dual | Native FP64 characteristic adjoints and dual/hyper-dual, search and roots on CUDA | Explicit CPU FP64 bridge; output MPS float32 |

MPS RF/surface support is an interoperability path, **not native GPU acceleration**.
It warns when selected and reports `backend='mps-cpu-fp64'` and CPU computation.
Input float32 rounding and output conversion remain; FP64 internal arithmetic
cannot recover precision already lost in an MPS input. CPU or CUDA float64 is
recommended when that distinction matters. All native custom backward paths
support first-order differentiation only.

## Minimal forward and gradient calls

Install the [dev7 binary preview](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev7), NumPy and a compatible
Torch distribution. RF/surface columns use top depth (km), Vp/Vs (km/s), density
(g/cm³); Green `Forward` uses SI and six columns including Qp/Qs.

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

device = 'cuda:0'  # 'cpu' or 'mps'
dtype = torch.float32 if device == 'mps' else torch.float64
one = [[0.,3.6,2.,2.5], [2.,6.5,3.6,2.8], [35.,8.,4.5,3.3]]
earth = torch.tensor(one, dtype=dtype, device=device)[None].repeat(16,1,1)
earth.requires_grad_()
rf = grtm.JointRFForward([.06], device=device, kernel='native', threads=4)
pred = rf(earth)
g_rf = torch.autograd.grad(pred['rf'].square().mean(), earth)[0]
sw = grtm.SurfaceWave(earth, threads=1 if device.startswith('cuda') else 4)
c = sw.phase_velocity([5.,10.,20.])
g_sw = torch.autograd.grad(c.sum(), earth)[0]
k = sw.sensitivity([5.,10.,20.], kind='group', parameter='velocity_s')['kernel']

model = grtm.example_model(units='si')
model.update(log2_samples=6, dt=.4, distances=[10000.,30000.], t0=[0.,0.], ptam=0)
backend = 'cuda' if device.startswith('cuda') else 'metal' if device == 'mps' else 'cpu'
f = Forward(model, backend=backend, device=0, parameters=['vp','vs'],
            workers=4, threads=4, vjp_method='auto')
x = torch.tensor(np.asarray(model['layers']), dtype=dtype, device=device)
x = x[None].repeat(16,1,1).requires_grad_()
mt = torch.tensor(grtm.strike_dip_rake(20,40,60,1e15), dtype=dtype, device=device)
y = f(x, mt)
g_green = torch.autograd.grad(y.square().mean(), x)[0]
print(pred['rf'].shape, c.shape, k.shape, y.shape)
```

This example selects gradients only for Green Vp/Vs. Other Green columns return
zero. RF/surface output uses batch × observation dimensions. `device=0` in the
Green constructor is the native CUDA card index; match it to the tensor device.
MPS warning and host computation are expected for RF/surface.

## Same-device boundaries

Green CUDA JVP propagates analytic derivatives through the original GRTM equations. `vjp_method='adjoint'` now uses true CUDA reverse AD: one shared graph per wavenumber/frequency and one backward pass for all selected structural parameters. `auto` chooses it when supported. CPU retains native reverse AD.

For CUDA tensors, analytic derivatives and PTAM=0, `grtm.torch.Forward` keeps spectra, FFT, radiation synthesis, strain/stress and source gradients on GPU by default. Small model/topology and fixed-grid planning remain on host. `device_pipeline=False` retains the old host-synthesis reference; NumPy/JAX callbacks still have host-array boundaries. PTAM JVP stays on host; native CUDA VJP requires PTAM=0.

The shared quasistatic criterion `|omega| < 5e-3 * vmax * nk * dk` retains host extended precision for those bins in forward, JVP and adjoint. CUDA backward transfers only their cotangents. This tightens dev6's 3e-3 guard to address derivative rounding near the transition. `Forward.execution_stats()` captures the most recent model backward's GPU/CPU frequency counts, parameter count and tape size on the autograd worker thread. `Solver.backend_stats()` remains native calling-thread statistics. Passing fixed validation cases does not certify every extreme long-period model.

RF uses native CPU kernels by default; `kernel='torch'` keeps the independent
Torch reference on CPU/CUDA. Native preparation uses analytic material adjoints. `threads` bounds native CPU workers; Torch FFT and other operations have their own `torch.set_num_threads()` budget. The legacy NumPy `synthetic_rf` is
a separate interface. CUDA RF reduces frequency cotangents within a warp before
writing gradient temporaries: the `gi`/`gbc` arrays have approximately 1/32 as many
frequency rows. The forward tape and total memory do not shrink by that factor.

Surface defaults to `differentiation="adjoint"`, reusing left/right states; `"forward"` retains the original analytic reference. Group kernels use analytic mixed derivatives with the original analytic path protecting critical coordinates. CPU parallelizes base evaluation and sensitivity over model × period;
root discovery retains guarded per-model continuation. Default tolerance remains
1e-10 km/s. CUDA performs numerical root search and derivatives on GPU; small host
validation/search-mesh preparation remains. No production finite differences or
disba dependency is introduced.

## Batches and memory

RF and surface have native CUDA model batching. Green CUDA tensors dispatch models sequentially and stack results on GPU; the legacy callback uses ordered host model workers. Neither fuses the model axis into one GPU kernel. `workers=4, threads=4` bounds native host work to 16 threads. Shared
source gradients reduce in model order. `workers=1` avoids a host thread pool.
CUDA Green workspaces reuse buffers and independent streams per host worker.
Tile scratch is targeted at 512 MiB per workspace; retained high-water buffers,
other entry-point caches and outputs are additional memory. This is not a total
process memory cap. Measure smaller batches before increasing concurrency.

See [measured timings and accuracy](PERFORMANCE_0.6.0.dev5.md), including default
disba and unchanged pyfk, and the [batch timing definitions](BATCH_EXECUTION.md).
Historical dev4 reports remain available and describe the older implementation.

See the [analytic-method guide](ANALYTIC_METHODS.md) for derivations, CPU Levin waveform/JVPs and validation scope.
