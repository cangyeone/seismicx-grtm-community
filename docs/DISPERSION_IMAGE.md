# Forced-response dispersion energy images

Available in **0.6.0.dev11** through `SurfaceWave.dispersion_image` or `surface_wave_image`. The elastic layered-medium solver applies a harmonic **surface traction**, solves the forced GRTM boundary system, and returns complex surface displacements on a frequency/phase-velocity grid. CPU and CUDA use the same FP64 formulas; CUDA fuses models and grid points in one native launch.

```python
import numpy as np
from grtm import SurfaceWave

# Native units: top depth km, Vp/Vs km/s, density g/cm³.
layers = [[0., 3.6, 2., 2.5], [2., 5.5, 3.1, 2.7], [10., 8., 4.5, 3.3]]
sw = SurfaceWave(layers, device='cpu', missing='nan')  # or 'cuda:0'
f = np.linspace(.025, .5, 121)                       # Hz, strictly increasing
c = np.linspace(1.7, 4.49, 701)                      # km/s, strictly increasing
image = sw.dispersion_image(f, c, wave='rayleigh')
# image['response']: [F,C,2], complex128, [radial, vertical]
# image['power']: [F,C], raw |u_vertical|²
# image['energy']: [F,C], power normalized independently at each frequency
# image['valid']: finite, numerically resolved response; NOT a mode-existence mask
phase = sw.phase_velocity(1/f, wave='rayleigh', mode=0)
```

`examples/dispersion_energy_image.py` saves PNG/PDF figures with phase curves overlaid, raw complex response/power/masks in NPZ, and metadata in JSON. Plot `image['energy'].T` against frequency and velocity. Logarithmic display floors belong in plotting, never in the saved raw arrays.

Minimal plotting after the example above (requires matplotlib):

```python
import matplotlib.pyplot as plt
# For CUDA outputs, move only the finished image to the host for plotting.
e = image['energy']
if hasattr(e, 'detach'):
    e = e.detach().cpu().numpy()
plt.pcolormesh(f, c, e.T, shading='auto', cmap='magma')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase velocity (km/s)')
plt.colorbar(label='Normalized displacement power')
plt.savefig('dispersion-image.png', dpi=180)
```

## Sources, components and units

- Rayleigh: `source='vertical'` by default; `'radial'` or complex `[radial, vertical]` amplitudes are also supported. Love: `wave='love'`, default `source='transverse'`, or `[complex_amplitude]`.
- `component` selects `'vertical'`, `'radial'` (Rayleigh), `'transverse'` (Love), or `'total'` for the sum of displacement powers. The full complex vector is always returned.
- The default source amplitude is one unit of **harmonic traction**: GPa in native units, Pa with `units='si'`. Response is km or m respectively; raw `power` is km² or m². Comparing SI and native requires converting the source too: 1 GPa = 10⁹ Pa. A unit-traction result is also the corresponding surface compliance column. This is a spectral plane-wave load; it is not a finite-offset point force in newtons.
- Use `[B,L,4/6]` models for batch; the output adds a leading B axis. Source amplitudes can be shared `[2/1]` or per model `[B,2/1]`. Both source and receiver are at depth zero.
- `normalization='frequency'` divides power by its maximum across velocities, independently for each model/frequency. `'global'` uses one maximum per model; `'none'` returns raw power as `energy`. All-zero rows stay zero. Failed samples remain NaN with `valid=False`; no missing-mode imputation occurs.

## Boundary solve and interpretation

At each real frequency and velocity, set

\[
 k=2\pi f/c,\qquad \widetilde\omega=2\pi f(1-i\eta),\qquad\eta>0.
\]

GRTM's modal eigenmatrices and interface continuity propagate the decaying/outgoing half-space subspace upward with bounded `exp(-q h)` factors. With surface displacement matrix U and traction matrix T, solve `T a = applied traction` (with the physical surface-normal convention), then evaluate `u = U a`. The free-surface modal condition is `det(T)=0`; the forced solution retains excitation and receiver-component weights. No inverse-determinant image is used. `boundary_residual` reports the relative residual of this small surface solve; it is not a full conditioning/error estimate.

`damping=0.002` is positive complex-frequency pole broadening. It controls peak width and height, and is **not** material attenuation/Q. Decreasing it requires a finer velocity grid; finite damping and coarse grids can move sampled maxima relative to undamped roots. Response power measures squared displacement amplitude, not volume-integrated elastic energy or energy flux. A mode can be weak or invisible for a particular excitation/component, and a smooth forced response can exist where no trapped mode exists. This spectral response is not an array MASW/F–J image: offset integration, aperture, source spectrum and processing would additionally affect an observed image.

For the impedance/free-boundary connection see [Jiang et al. (2025), equations 3–5](https://link.springer.com/article/10.1186/s40623-025-02164-2). The implementation uses this project's existing GRTM eigen/interface maps.

## Scope and verification

Elastic, isotropic solid layers over a radiating half-space; optional Q columns are ignored consistently with the existing surface-wave solver. No fluids, material attenuation, buried source/receiver, or image gradients in this release. Trainable tensor inputs raise an explicit error unless called under `torch.no_grad()`; dispersion-curve gradients and sensitivity kernels remain available through their existing interfaces.

CPU returns NumPy, or Torch tensors when given Torch models. CUDA keeps the numerical solve, response and normalization on its selected device/current stream; input checks may synchronize scalar validation results. MPS remains an explicit CPU FP64 bridge: complex response/source stay on CPU, real images and masks return on MPS (float32). Metadata records this distinction.

Tests compare with an independently assembled global P/SV wave equation, analytic SH half-space/single-layer solutions, layer splitting, source linearity, density scaling, SI conversion, modal peak positions, batch and nondefault CUDA streams. Accuracy and timing records accompany the release; CPU/CUDA agreement alone is not the physical validation.
