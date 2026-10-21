# Differentiable receiver functions on CPU (0.6.0.dev2 preview)

**Historical dev2 documentation.** The current [dev4 preview](PREVIEW_0.6.0.dev4.md)
also supports native CUDA forward/VJP. Read the [current batch examples and execution guide](BATCH_EXECUTION.md)
and [2026-10-21 throughput measurements](BATCH_SCALING_20261021.md). CPU-only restrictions
below describe dev2, not dev4.

This preview adds the public CPU interfaces `JointRFForward`, `joint_receiver_forward`, `hmc_sample`, `rank_diagnostics`, `dram_sample`, `chain_diagnostics`, `ShallowCrustModel`, and `JointRFProblem`. The previous published wheel did not expose the project's differentiable RF/HMC implementation.

The RF interface uses CPU PyTorch float64/complex128 scattering and FFT operations. Ordinary first-order PyTorch autograd differentiates model depths, velocities, density and finite Q. Production HMC uses first derivatives; the preview does not claim higher-order derivative certification or universal model accuracy. The existing Green-function binaries from 0.6.0.dev1 remain unchanged. The newly added RF modules contain no accelerator execution path and reject devices other than CPU.

## Install the matching community wheel

[Release 0.6.0.dev2](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev2) provides macOS arm64 / CPython 3.12 and Linux x86_64 / CPython 3.13 (manylinux_2_28) wheels. Check `SHA256SUMS-0.6.0.dev2` before installation. Other Python versions retain the earlier release; they do not have this new CPU RF interface yet.

```bash
python -m pip install numpy scipy torch
python -m pip install /path/to/matching/seismicx_grtm-0.6.0.dev2-....whl
```

Linux users who want only CPU PyTorch can install it first with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```python
import torch
from grtm import JointRFForward

layers = torch.tensor([[0., 3.6, 2., 2.5],
                       [2., 6.5, 3.6, 2.8],
                       [35., 8., 4.5, 3.3]], dtype=torch.float64,
                      requires_grad=True)
op = JointRFForward([.045, .065], gaussians=[2.5], periods=[0.],
                    dt=.2, n_samples=151, t0=-2., nfft=512, device='cpu')
result = op(layers)
rf = result['rf']                 # [batch, ray, band, time]
gradient = torch.autograd.grad(rf.square().sum(), layers)[0]
print(rf.shape, gradient)
```

Rows use top-depth km, Vp/Vs km/s, density g/cm³, and optionally Qp/Qs. The last row is the halfspace. Four-column models are elastic; six-column models support `attenuation='constant_q'` with positive Q or infinity. Tops must begin at zero and increase; rays must be subcritical; Vp must exceed sqrt(4/3) Vs. These are isotropic solid-layer, 1D plane-P synthetics, including reverberations. Water-level normalization by the vertical autocorrelation is part of the solver. Radial normalization at t=0 for comparison to particular archived RF products is a caller-level processing convention; apply it consistently to synthetics and observations.

`hmc_sample(log_prob, initial, lower, upper, device='cpu', steps=..., burn_in=...)` samples a differentiable physical-coordinate box target. It adds the logit box Jacobian, keeps rejected states, adapts only during warmup and reports divergences. `dram_sample` is our independent implementation of delayed-rejection adaptive Metropolis, not redistributed third-party inversion source. Use `rank_diagnostics` on `[production_draw,chain,parameter]`; short or constant chains are not convergence evidence.

For a ready RF inversion API, models and reproduction materials use [FlowRF](https://github.com/cangyeone/flow-assisted-hmc-receiver-function-inversion). FlowRF installs GRTM separately and does not redistribute this solver's implementation. Third-party Bayesian comparison programs are not included in either new RF preview material.

## License / validation

The existing SeismicX GRTM research and noncommercial license remains in effect. This release distributes binary modules, not solver source. The companion [Chinese instructions](DIFFERENTIABLE_RF_CPU.zh-CN.md) and CPU verification reports describe supported environments and numerical checks. Original Green-function binaries preserve their previous notices and behavior.
