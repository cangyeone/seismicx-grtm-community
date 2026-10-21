# Oscillatory integration and analytic adjoints in GRTM

Version 0.6.0.dev7 optimizes forward modelling and derivatives using the same GRTM physics. Production derivatives do not perturb models and repeat root searches. Finite differences are independent validation tools. Supported parameters include interfaces or thicknesses, Vp, Vs, density and, for attenuating responses, Qp/Qs.

## A shared differentiation principle

Surface waves satisfy the characteristic equation `F(c,T,m)=0`, giving `c_m=-F_m/F_c`. The new implementation stores right propagation states and left adjoints and reuses them for all layer parameters. Away from protected modal coordinates, the cost of all kernels changes from O(L²) to O(L). This is a complexity statement, not a promise about complete API speed. Critical coordinates retain the original analytic path for accuracy. Group velocity is `U=c²/(c+T*c_T)`; its kernels use analytic mixed derivatives along the root path `(dc/dT,dT)=(c_T,1)`, without period differences.

Receiver functions and Green's functions are forced responses, so the surface-wave eigenvalue energy formula cannot simply be reused. The common principle is differentiation of the governing equations. For `AX=B`, `dX=A⁻¹(dB-dA X)`; solving `AᴴQ=bar_X` gives `bar_B=Q` and `bar_A=-QXᴴ`. Native RF preparation uses these matrix adjoints and analytic material derivatives. Green's functions apply analytic reverse chain rules to complex operators generated from the shared GRTM source. Complex cotangents follow `dL=Re(sum(conj(bar_x)*dx))`. The analytic characteristic equation uses the corresponding bilinear transpose convention, with conjugation handled explicitly.

[QEDispInv](https://github.com/pan3rock/QEDispInv) motivated examining eigenfunction kernels; its interface calls CPS `sregn96/slegn96`. This implementation independently derives FP64 GRTM characteristic adjoints and does not copy those CPS kernels or claim identical implementation. The [CPS group-kernel tutorial](https://rbherrmann.github.io/ComputerProgramsSeismology/TUTORIAL/SWKERNELS/index.html) describes numerical period derivatives; GRTM uses analytic mixed derivatives here.

## Where Levin integration applies

Green's wavenumber integrals contain `J0(kr)` and `J1(kr)`. Levin integration uses the Bessel differential equation to solve an auxiliary equation and evaluate the integral at panel endpoints. Batched sampling and cached, real geometry-only SVD factorizations reduce overhead. Forward fields and analytic JVPs share adaptive panels. Gauss can be preferable in the near field; `method='auto'` selects using the oscillatory span. This is a Levin implementation; no additional Filon implementation is claimed.

Dispersion roots and the current plane-P receiver functions do not contain this wavenumber integral. Their optimizations target characteristic and scattering equations, respectively; RF frequency-to-time conversion uses FFT. Levin does not replace FFT or general root finding.

References: [Levin's oscillatory integration analysis](https://www.tau.ac.il/~levin/OscIntAnal.pdf), [adaptive Levin integration](https://arxiv.org/abs/2211.13400). Finite cutoffs, sampled residuals and embedded error estimates do not rigorously bound the infinite tail or native floating-point error.

## Interfaces

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

layers = [[0., 3.6, 2., 2.5], [2., 6.5, 3.6, 2.8], [35., 8., 4.5, 3.3]]
sw = grtm.SurfaceWave(layers, device='cuda:0', differentiation='adjoint')
phase = sw.sensitivity([3., 8., 18.], kind='phase', parameter='all')
group = sw.sensitivity([3., 8., 18.], kind='group', parameter='all')
# differentiation='forward' selects the original analytic reference, not FD.

x = torch.tensor(layers, dtype=torch.float64, device='cuda:0', requires_grad=True)
rf = grtm.JointRFForward([.04, .06], device='cuda:0', kernel='native')
rf_gradient, = torch.autograd.grad(rf(x)['rf'].square().sum(), x)

model_si = grtm.example_model(units='si')
model_si.update(log2_samples=5, dt=.4, distances=[10000.], t0=[0.], ptam=0)
green = Forward(model_si, backend='cuda', parameters=['vp', 'vs', 'rho'],
                vjp_method='adjoint')
x_si = torch.tensor(model_si['layers'], dtype=torch.float64,
                    device='cuda:0', requires_grad=True)
mt = torch.tensor(grtm.strike_dip_rake(20, 40, 60, 1e15),
                  dtype=torch.float64, device='cuda:0')
waveform = green(x_si, mt)
gradient, = torch.autograd.grad(waveform.square().sum(), x_si)
print(green.execution_stats())

direction = np.zeros_like(np.asarray(model_si['layers'], dtype=float))
direction[:, 2] = 1.  # Vs direction, m/s
joint = grtm.Solver().forward_quadrature(
    model_si, cutoff=4., method='levin', direction=direction,
    moment_tensor=mt.detach().cpu().numpy(), fields=('displacement',),
    atol=1e-10, rtol=0.)
y, dy = joint['displacement'], joint['jvp']['displacement']
```

Surface/RF layer columns above are top depth in km, Vp/Vs in km/s and density in g/cm³. Green's configuration and direction use SI. Wavenumber cutoff always uses km⁻¹. Quadrature tolerances apply to raw integration channels, not guaranteed relative errors of displacement or stress. JVPs differentiate the integrand, not discrete panel-selection branches. Explicit `subtract_direct=True` changes the target to an analytic Abel direct field plus finite-cutoff scattered field.

## Device and validation boundaries

- Surface CPU/CUDA roots and kernels, and RF native preparation/propagation, run on the selected device. Native framework backward interfaces guarantee first-order derivatives only.
- With CUDA input tensors, analytic derivatives and `PTAM=0`, `Forward` keeps spectra, FFT, radiation synthesis, strain/stress, source gradients and structural reverse AD on GPU. Small model/topology and fixed-grid planning remain on the host. `device_pipeline=False` selects the old host-synthesis reference. NumPy interfaces still return host arrays.
- Nearly dependent dynamic P/S coordinates at low frequency require extended-precision protection when `|omega|/(vmax*kmax)<5e-3`. Those bins retain host calculation; CUDA backward transfers only their spectral cotangents. `execution_stats()` captures the most recent model backward's counters on its autograd worker thread. Green's model batch is not a fused native batch kernel.
- Adaptive Levin/Gauss and joint waveform JVPs currently run on CPU. RF/surface MPS is an explicit CPU FP64 bridge with float32 real outputs. Green Metal retains its existing hybrid implementation.
- Validation covers higher-mode root-recomputed differences, physical scaling identities, an independent RF tensor graph, Green JVP/VJP duality, all physical fields against CPU, nondefault CUDA streams, shared-source batches, parameter masks and low-frequency protection. Higher-mode disba sensitivities are not treated as derivative ground truth.

Report known-root kernel timings, full API timings, single models and batches separately. Complexity and prototype timings cannot substitute for measured end-to-end speed.
