# Model batches, CUDA execution and throughput

English | [简体中文](BATCH_EXECUTION.zh-CN.md)

These interfaces are in the [0.6.0.dev4 compiled preview](PREVIEW_0.6.0.dev4.md).
Stable PyPI is 0.5.1. Read the [batch-scaling measurements](BATCH_SCALING_20261021.md)
for model counts, CPU budgets, latency, models/s and numerical checks.

| Interface | Meaning of a model batch | Structural derivatives |
|---|---|---|
| `JointRFForward` | Native CUDA model × ray × frequency work | Native CUDA VJP plus Torch CUDA preparation/FFT; CPU Torch reference available |
| `SurfaceWave` | Native CUDA model/period/search/parameter tasks | Native CUDA dual/hyper-dual kernels; CPU model workers via `threads` |
| `Solver.forward_batch` | Host pool schedules separate Green-function calls | Each model retains its own forward allocation and transfer |
| `ForwardOperator` batched input | Sequential independent model calls | Green JVP/VJP remains CPU, including CUDA-base pipelines |

The Green solver does parallelize frequencies/wavenumbers/receivers inside one
model. The additional model batch is not fused into a single CUDA launch.
`workers=4` is host scheduling, not four fused GPU models or four GPUs.

## Surface-wave models

```python
import numpy as np
import torch
from grtm import SurfaceWave

# Top depth km, Vp/Vs km/s, density g/cm³; final row is a halfspace.
one = np.array([[0.,3.2,1.65,2.1], [.8,5.6,3.15,2.6],
                [5.,6.7,3.8,2.9], [25.,8.1,4.6,3.3]])
models = np.repeat(one[None], 64, axis=0)  # [B,L,4]
earth = torch.tensor(models, dtype=torch.float64, device='cuda:0')
sw = SurfaceWave(earth)  # CUDA threads=1 (the default)
periods = np.geomspace(2, 50, 12)
c = sw.phase_velocity(periods)  # [64,12]
k = sw.sensitivity(periods, kind='group', parameter='velocity_s')['kernel']
print(k.shape)  # [64,12,4]

cpu = SurfaceWave(models, threads=16)  # explicit CPU model workers
c_cpu = cpu.phase_velocity(periods)
```

`threads=16` is for CPU, not CUDA. Native FP64/complex128 and the 1e-10 km/s
default root tolerance are unchanged by batch size. Models use top depths by
default; set `parameterization='thickness'` for thickness arrays. CPU NumPy
outputs and CUDA tensors have the same shape conventions.

Tensor velocities support first-order model autograd. Explicit `sensitivity()`
returns derivative data and is not itself a higher-order autograd graph.
See [surface-wave conventions and limitations](SURFACE_WAVES.md).

## Receiver functions and gradients

```python
import torch
from grtm import JointRFForward

# Example geometry only; the published scaling benchmark uses 32 layers.
one = torch.tensor([[0.,3.6,2.,2.5], [2.,6.5,3.6,2.8],
                    [35.,8.,4.5,3.3]], dtype=torch.float64, device='cuda:0')
earth = one[None].repeat(64,1,1).requires_grad_()
op = JointRFForward([.06], device='cuda:0')
pred = op(earth)
rf = pred['rf']              # [64,1,6,81]
vs_app = pred['vs_app']      # [64,1,6,41]
gradient = torch.autograd.grad(rf.square().mean(), earth)[0]  # [64,3,4]
```

Columns are top depth, Vp, Vs, density in native km/km/s/g/cm³; the first top is
fixed and the final layer is a halfspace. CUDA propagation/backward and Torch
preparation/FFT remain on GPU. `device='cpu'` selects the CPU Torch reference;
set its global Torch thread count explicitly when benchmarking. CPU 4 threads
and CPU 16 model workers in other interfaces are different resource settings.
The legacy NumPy `synthetic_rf` API is a separate CPU interface.

## Green-function host batches

```python
import numpy as np
import grtm

model = grtm.example_model(units='si')
model.update(log2_samples=6, dt=.4, distances=[10000.,30000.], t0=[0.,0.])
base = np.asarray(model['layers'])
models = []
for factor in np.linspace(.99,1.01,16):
    layers = base.copy()
    layers[:,2] *= factor
    models.append(dict(model, layers=layers))
moment = grtm.strike_dip_rake(20,40,60,1e15)
solver = grtm.Solver(backend='cuda')
out = solver.forward_batch(models, moment_tensor=moment,
                           fields=('displacement',), workers=4, threads=4, device=0)
print(out['displacement'].shape)  # [16,2,64,3]

# The base forward is CUDA; this structural VJP is CPU.
op = solver.operator(model, parameters=['vp','vs'], threads=16,
                     device=0, vjp_method='adjoint')
linear = op.linearize(base, moment)
gradient, _ = linear.vjp(np.ones_like(linear.value), moment_gradient=False)
```

Here 4 host model workers × 4 native CPU threads give a 16-thread CPU budget.
Model I/O and GPU allocations/transfers remain per call. Calling
`op(earth_batch, moment)` accepts a batch but currently loops models sequentially;
it does not use `forward_batch`'s worker pool. Structural VJP does not gain a
CUDA model axis from either interface. See the [forward/derivative guide](PYTHON_FORWARD.md).

## Measure latency and throughput

```python
import statistics
import time

def seconds(function, device, repeat=3):
    function()  # warm up libraries and the selected shape
    torch.cuda.synchronize(device)
    trials = []
    for _ in range(repeat):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        trials.append(time.perf_counter()-start)
    return statistics.median(trials)

t = seconds(lambda: sw.phase_velocity(periods), 'cuda:0')
batch = 64  # the SurfaceWave example above
print(batch/t, 'models/s;', 1000*t/batch, 'ms/model')
```

The snippets are independent examples; retain the matching device/model objects
when timing one calculation. For a batch B taking T seconds:

- Whole-batch latency: T; mean milliseconds/model: `1000*T/B`.
- Throughput: `B/T` models/s.
- CUDA versus CPU at the same batch: `T_CPU(B)/T_CUDA(B)`.
- GPU throughput gain over B1: `B*T_CUDA(1)/T_CUDA(B)`.

Reuse/resident-model timings exclude first construction and upload. In an
inversion that changes models every step, also measure the complete step with
model generation, any search-mesh refresh and transfer. Public validation and
host orchestration still matter at small B. Reverse RF memory grows with model
count, layers and FFT frequencies; choose batch size from measured throughput
and memory rather than assuming the largest batch is fastest. Batch size never
changes the advertised precision or the current CPU/GPU derivative boundary.

The four Python examples above were executed on an installed dev4 wheel and
RTX 5090 on 2026-10-21; English/Chinese Python syntax trees match. See the
[execution record](../verification/batch-scaling-20261021/documentation-examples-20261021.json).
