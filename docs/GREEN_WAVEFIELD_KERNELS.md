# Explicit Green wavefield integral kernels (dev9)

Select `Forward(..., vjp_method='wavefield')` or `ForwardOperator(..., vjp_method='wavefield')`. For raw spectra use `Solver.spectrum_vjp(..., method='wavefield')`. CPU and CUDA support FP64 batches, force and moment sources, displacement/gradient/strain/stress, both signal and spectrum outputs, and all six original model columns. Defaults remain unchanged; `vjp_method='adjoint'` retains the reverse AD reference. `Forward.sensitivity` preserves an explicit wavefield selection.

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

device = 'cuda'  # or 'cpu'
model = grtm.example_model(units='si')
model.update(source_depth=8000., receiver_depth=0., ptam=0,
             log2_samples=7, dt=.1, distances=[10000.], t0=[0.])
op = Forward(model, backend=device, vjp_method='wavefield', threads=4,
             parameters=['depth','vp','vs','rho','qp','qs'])
x = torch.tensor(np.asarray(model['layers']), dtype=torch.float64,
                 device=device, requires_grad=True)
source = torch.tensor(grtm.strike_dip_rake(20,40,60,1e15),
                      dtype=x.dtype, device=device)
y = op(x, source)
gradient, = torch.autograd.grad(y.square().sum(), x)
print(op.execution_stats())
```

This is a scalar-observable VJP, not a dense sample-by-parameter Jacobian. There are no full-stack parameter sweeps; work still scales with layers and integration nodes.

The shared implementation reconstructs bounded, top/bottom-anchored P-SV/SH modes using the existing GRTM constitutive law and solves their exact block-tridiagonal dynamic-stiffness representation. It integrates forward/adjoint modal pairs analytically using exponential divided differences and the convergent halfspace tail. It includes source jumps, source normalization, receiver derivative operators, interface shape terms, attenuation, and existing physical synthesis derivatives. No depth quadrature or material finite differences are introduced. With state `y=(u,t)` and bilinear adjoint state `p=(-t_v,v)`, the material contraction is `-Re ∫ pᵀ δA y dz`.

For dissipative GRTM, `μ=ρ Vs²`, `C=μ a²/b²`, and `D=μ ω²/b²`; D/ω² is the effective inertia of this convention. Treating it as unmodified real density would change the existing solver. Three contractions for μ,C,D map analytically to Vp,Vs,ρ,Qp,Qs. See the [full derivation](GREEN_WAVEFIELD_KERNELS.zh-CN.md).

Accuracy guards retain reverse AD for the existing quasistatic frequencies (unchanged threshold 5e-3), coincident source/receiver depths or a surface source, sublayers thinner than 1e-4 km, and singular/nonfinite layer solves. Regular CUDA fallback remains on CUDA. Quasistatic bins retain the existing extended-precision CPU exception; diagnostics disclose it. CPU arithmetic is not reduced to float32. No integration counts or tolerances are relaxed.

CUDA distributes modal preparation and overlaps over frequency × wavenumber × layer, interleaved with small node-wise solves and deterministic reductions. Spectra, synthesis, FFT and regular structural gradients stay on the selected GPU. The batch API currently schedules models separately; it does not fuse different models into one native launch. Large windows can remain dominated by protected CPU bins; disclose their count and CPU thread setting when reporting speedups.

QEDispInv's [`kernel(freq,c)`](https://github.com/pan3rock/QEDispInv/blob/ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa/src/swegn96.cc) evaluates a known-root surface-mode kernel, not a full Green wavenumber integral. Its model input is float32 but most Fortran internals are float64. The matched comparison uses identical known roots and excludes root searches on both sides. Its specialized real formulas remain faster than our generic complex GRTM energy formulas. Dev9 removes the unused Fc/Ft traversal for phase energy kernels; group-period derivatives remain intact. No QEDispInv solver code was copied into this implementation.
