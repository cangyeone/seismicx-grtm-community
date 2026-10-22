# Eigenfunction and forced-response kernels

[中文：完整推导与接口](VARIATIONAL_KERNELS.zh-CN.md)

dev8 adds `SurfaceWave(differentiation='eigenfunction')`: FP64 GRTM mode reconstruction, closed-form exponential energy integrals, interface shape derivatives and an analytic root-path period derivative for group kernels. It also adds `JointRFForward.sensitivity` and `grtm.torch.Forward.sensitivity` for scalar-observable adjoint kernels.

These implementations are distinct. Surface waves explicitly integrate eigenfunction energies. RF uses scattering matrix adjoints and a new constitutive-kernel contraction; Green functions retain the verified analytic complex reverse operators. The latter two implement the discrete forced-response variational identity; **a separate continuous-depth volume-integral backend has not been added**. They are not renamed eigenfunction integrals.

For `D(k,ω,m)u=[K(k,m)−ω²M(m)]u=0`, at fixed frequency,

\[
c_m=\frac{c}{k}\frac{u^H D_m u}{u^H D_k u}.
\]

The denominator is `uᴴD_k u=2ω U∫ρ|u|² dz`: flux/energy normalization removes arbitrary mode amplitude.

GRTM downward waves are anchored at layer tops and upward waves at layer bottoms. Exponential products are integrated analytically, with a series for nearly zero exponent sums. No depth quadrature or model perturbation is used. The three material numerators are `2ρVp Idiv`, `2ρVs(Istrain−2Idiv)` and `(Vp²−2Vs²)Idiv+Vs²Istrain−ω²Iu`, with `Istrain=∫2 ε:ε*`. Thickness kernels use Hamiltonian jumps at interfaces. Half-space thickness and Love Vp kernels are exactly zero.

For group velocity `U=c²/(c+T c_T)`, a dual-number pass along `(dc/dT,dT)=(c_T,1)` differentiates the eigenfunction, integrals and normalization to obtain `d(c_m)/dT`; the chain rule yields `U_m`. There is no period finite difference or full propagation per material parameter.

The returned `kernel_path` is 0 for energy integration, 1 for the same-device analytical reference fallback, and 2 for a missing root. `energy_residual` and `boundary_residual` include dimensionless root-path derivative residuals for group kernels and check the selected solution, not mode completeness or numbering. Critical modal coordinates retain an explicit accuracy fallback. The default remains `adjoint`; the energy method is not assumed faster for every workload. This new energy path is limited to isotropic elastic solid layers.

For the forced equation `A(m)u=f(m)` and a real scalar objective, define `δJ=Re(gᴴδu)+J_m δm`. Then

\[
A^H v=g,\qquad J_m^{total}=\operatorname{Re}\{v^H(f_m-A_m u)\}+J_m.
\]

In the elastic weak form its volume contractions are `Kλ=−Re∫div(v)* div(u)`, `Kμ=−Re∫2ε(v)*:ε(u)` and `Kρ=Re∫ω²v*·u`, plus interface/source/explicit observation terms. Discrete GRTM continuity solves use `X=A⁻¹B`, `Q=A⁻ᴴ bar_X`, `bar_B=Q`, `bar_A=−QXᴴ`. RF first contracts Lamé and density kernels, then maps analytically to Vp/Vs/Q, replacing its local parameter-direction loop. Green retains analytic reverse primitives, direct stress terms and source derivatives. Existing supported attenuation models are differentiated through their complex operators; this does not extend the elastic surface-energy formula to attenuating modes, fluids or anisotropy.

RF adjoint sources include both radial and vertical spectra, the water-level denominator and reference maximum, normalization, Gaussian filters, FFT and polarization. At max/tie switches the framework's subgradient convention applies. Freezing the vertical response would give an incorrect kernel.

```python
import torch
import grtm
from grtm.torch import Forward

layers = [[0.,3.6,2.,2.5], [2.,6.5,3.6,2.8], [35.,8.,4.5,3.3]]
sw = grtm.SurfaceWave(layers, device='cuda:0', differentiation='eigenfunction')
kernel = sw.sensitivity([3.,8.,18.], kind='group', parameter='all')

rf = grtm.JointRFForward([.04,.06], device='cuda:0')
value = rf(layers)
r = rf.sensitivity(layers, observed={'rf': torch.zeros_like(value['rf'])})
# r['kernel'] is d(half squared residual sum)/d(layers).

model = grtm.example_model(units='si')
model.update(ptam=0, log2_samples=5, dt=.4, distances=[10000.], t0=[0.])
green = Forward(model, backend='cuda', parameters=['vp','vs','rho'])
x = torch.tensor(model['layers'], dtype=torch.float64, device='cuda:0')
mt = torch.tensor(grtm.strike_dip_rake(20,40,60,1e15), dtype=x.dtype, device=x.device)
value = green(x, mt)
g = green.sensitivity(x, mt, cotangent=torch.ones_like(value))
```

Use either `observed` (half weighted squared residuals) or `cotangent` (real conjugate inner product). RF accepts a dictionary selecting `rf`, `zrf`, `vs_app`, `radial_spectrum` and/or `vertical_spectrum`. Weights may broadcast; data and cotangents must exactly match predictions. Outputs are detached first derivatives, preserving input `.grad`. Green forces the analytical adjoint without mutating the original operator and requires PTAM=0. Keep the integration grid fixed for finite-difference validation.

One scalar objective requires no parameter sweep. `reverse_passes=1` means one objective reverse traversal, still partitioned by Earth model, frequency, wavenumber and precision safeguards. Forming all layer kernels costs at least O(L), and a full sample-by-parameter Jacobian is not obtained for constant cost. Control-point gradients follow from `m=Bθ`, `gθ=Bᵀgm` without per-control wavefield solves.

CPU/CUDA use FP64. CUDA propagation and gradients remain on device except existing protected Green quasistatic bins. RF/surface MPS use the explicitly reported CPU FP64 bridge; the new Green interface requires CPU/CUDA tensors.

References: [QEDispInv](https://github.com/pan3rock/QEDispInv), [CPS kernel tutorial](https://rbherrmann.github.io/ComputerProgramsSeismology/TUTORIAL/SWKERNELS/index.html), [de Jong et al.](https://doi.org/10.1093/gji/ggac098), [Liu & Tromp](https://doi.org/10.1111/j.1365-246X.2008.03798.x). No CPS/QEDispInv implementation source is copied.
