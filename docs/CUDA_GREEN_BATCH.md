# Fused Earth-model Green batches (0.6.0.dev10)

[中文](CUDA_GREEN_BATCH.zh-CN.md) · [Measurements](PERFORMANCE_0.6.0.dev10.md)

CUDA `grtm.torch.Forward(..., vjp_method="wavefield")` now fuses models,
frequencies and wavenumbers into shared launches. Layer contractions also
parallelize over layers. Physical radiation, coordinate transforms, stress,
source-time convolution, FFT and their Torch derivatives operate on the whole
batch on the caller's GPU. This computes scalar-loss VJPs for all selected
parameters, not a dense waveform Jacobian.

```python
import numpy as np
import torch, grtm
from grtm.torch import Forward

model = grtm.example_model()
model.update(log2_samples=7, dt=0.4, ptam=0)
layers = torch.tensor(np.repeat(np.asarray(model["layers"])[None], 16, axis=0),
                      dtype=torch.float64, device="cuda:0", requires_grad=True)
moment = torch.tensor(grtm.strike_dip_rake(20, 40, 60, 1.0),
                      dtype=layers.dtype, device=layers.device)
forward = Forward(model, units="native", backend="cuda", threads=4,
                  vjp_method="wavefield", cuda_batch=True,
                  cuda_precision="fast",
                  grid={"steps": [512] * 64, "period_length": 500.0})
waveforms = forward(layers, moment)
gradient, = torch.autograd.grad(waveforms.square().sum(), layers)
print(forward.execution_stats())
```

`cuda_batch=True` is the wavefield tensor-path default. `[L,6]` and `[B,L,6]`
inputs are supported, with a shared `[6]` or per-model `[B,6]` moment tensor.
Original layer counts agree within a batch; depths, materials and source/receiver
expanded topology may differ. Fixed model configuration (sampling, receivers,
source depth, field and units) is shared. Input tensors must share device and
dtype. Internals remain FP64/complex128 even in fast mode. `cuda_batch=False`
retains serial model calls for controlled comparisons. CPU, MPS, RF and dispersion
algorithms are unchanged by this release.

## Precision policy and device boundary

The default `cuda_precision="strict"` retains the original low-frequency
extended-precision CPU guard. It gives fusion without intentionally relaxing
the numerical policy. `"fast"` is an explicit option for analytic PTAM=0 CUDA
wavefield calls with CUDA inputs. It extends the explicit FP64 layer-overlap kernels to calibrated low-frequency
bins; extreme cancellation retains the original protection. Geometry or
nonfinite-overlap fallback uses the original reverse AD on the GPU.

Define `r_f = |omega_f| / (Vp_max * K_f * dk)`, including imaginary-frequency
damping. Strict mode protects bins with `r_f < 0.005`. Fast mode retains that
entire policy for a model if any bin has `r_f < 0.0005`. Otherwise it protects
DC below `0.0005` and nonzero frequencies below `0.002`; newly admitted bins
below `0.005` use the same explicit wavefield integrals. These are empirical conditioning guards,
not a universal error certificate. Acceptance tests target relative waveform
error ≤1e-4 and model/source gradient error ≤1e-3, also checking each structural
parameter column separately. See the actual audited cases in the report.

Extremely long windows remain protected: tests found that small individual
frequency errors can produce larger errors in the final combined VJP. Very thin
sublayers, coincident source/receiver traces and failed overlap conditioning
use the original reverse convention per model. Such fallback can reduce speed.

All regular numerical arrays, GPU low-frequency kernels, synthesis and FFT stay on
the selected GPU. Model validation, topology, grid planning and small execution
counters are host work. One model batch snapshot crosses to the host per
forward call. If protected CPU bins remain, their spectra and gradients cross
the device boundary; backward currently downloads the batch cotangent once
for those bins. The implementation therefore claims full GPU numerical
execution only when `cpu_frequencies == 0`, not zero host API overhead.

`execution_stats()` is captured on the autograd worker after backward. Fused
counts cover the **entire batch**, including `models`, `cpu_frequencies`,
`cuda_frequencies`, `gpu_low_frequency_frequencies`, `strict_guarded_models`,
`reverse_frequencies` and `fallback_models`. Legacy serial calls report the
most recent model. Plans retain host metadata only; GPU scratch is cached per
worker/device with bounded stage budgets. Reductions are deterministic without
cross-model gradient atomics. Non-default caller streams and repeated retained
backward calls are covered by regression tests. Second derivatives remain
unsupported.
