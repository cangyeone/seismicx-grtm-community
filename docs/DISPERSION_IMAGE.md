# Forced-response dispersion energy images

Available since **0.6.0.dev11**, extended with analytic adjoints, PSD and finite arrays in **0.6.0.dev12** through `SurfaceWave.dispersion_image` or `surface_wave_image`. The elastic layered-medium solver applies a harmonic **surface traction**, solves the forced GRTM boundary system, and returns complex surface displacements on a frequency/phase-velocity grid. CPU and CUDA use the same FP64 formulas; CUDA fuses models and grid points in one native launch.

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

`damping=0.002` is positive complex-frequency pole broadening. It controls peak width and height, and is **not** material attenuation/Q. Decreasing it requires a finer velocity grid; finite damping and coarse grids can move sampled maxima relative to undamped roots. Response power measures squared displacement amplitude, not volume-integrated elastic energy or energy flux. A mode can be weak or invisible for a particular excitation/component, and a smooth forced response can exist where no trapped mode exists. The default direct spectrum excludes finite aperture; optional array operators below synthesize and process the spatial observations.

For the impedance/free-boundary connection see [Jiang et al. (2025), equations 3–5](https://link.springer.com/article/10.1186/s40623-025-02164-2). The implementation uses this project's existing GRTM eigen/interface maps.

## Scope and verification

Elastic, isotropic solid layers over a radiating half-space; optional Q columns are ignored consistently with the existing surface-wave solver. No fluids, material attenuation or buried source/receiver. First-order image adjoints are available on CPU/CUDA as described below; existing dispersion-curve derivatives are unchanged.

CPU returns NumPy, or Torch tensors when given Torch models. CUDA keeps the numerical solve, response and normalization on its selected device/current stream; input checks may synchronize scalar validation results. MPS remains an explicit CPU FP64 bridge: complex response/source stay on CPU, real images and masks return on MPS (float32). Metadata records this distinction.

Tests compare with an independently assembled global P/SV wave equation, analytic SH half-space/single-layer solutions, layer splitting, source linearity, density scaling, SI conversion, modal peak positions, batch and nondefault CUDA streams. Accuracy and timing records accompany the release; CPU/CUDA agreement alone is not the physical validation.

## Finite arrays and source PSD (dev12)

`source_psd` accepts a nonnegative scalar, `[F]` or `[B,F]`. It multiplies squared reference-source response; complex `response` remains the unweighted transfer response. When supplied, PSD describes a dimensionless reference-source time function in 1/Hz and `power` is displacement²/Hz; calibrated force PSD is `|source|² * source_psd`. Default `None` is unit weight. Positive per-frequency PSD cancels under `normalization='frequency'`; invert PSD with raw `power`. Common source amplitude also cancels from normalized images. Normalization, absolute values and validity masks are only piecewise differentiable.

```python
from grtm import DispersionArray

# One reusable plan per geometry, for repeated inversion calls.
line = DispersionArray(method='fk', field='coherent', length=1.07,
                       n_receivers=108, offset=.05, window='hann',
                       integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
line_image = sw.dispersion_image(f, c, array=line, source_psd=np.ones(len(f)))

# Active vertical point source -> vertical records at radial distances.
active = DispersionArray(method='fj', field='coherent',
                         radii=np.linspace(.05, 1.12, 108),
                         integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
active_image = sw.dispersion_image(f, c, array=active)

# Isotropic random surface forcing -> station-pair covariance -> FJ.
noise = DispersionArray(method='fj', field='noise',
                        positions=np.linspace(0., 1.07, 108),
                        integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
noise_image = sw.dispersion_image(f, c, array=noise)
```

The integration grid above is illustrative: converge its spacing **and endpoints** for each model/frequency/damping/aperture. `array=None` retains the original direct spectrum. Array results retain direct `response`, `direct_power`, `direct_valid` and additionally return `array_response`, `receiver_response` when applicable, processed `power/energy/valid`, distances, window, integration grid and `integration_phase_step_max = max(Δk)*max(|r|)`. A large phase step indicates spatial oscillations are underresolved; a small one does not guarantee resolving narrow poles. All synthesis is explicitly **band limited** to the supplied positive-k interval. The default uses the display velocity grid. There is no extrapolation to zero/infinite k, nor an automatic convergence certificate.

| Observation | Source units, native / SI | Receiver and array response | Display power |
|---|---|---|---|
| Direct surface spectrum | GPa / Pa | km / m | squared displacement |
| Coherent line-array FK | GPa·km / N/m | complex displacement | squared beam displacement |
| Coherent point-source FJ | GPa·km² / N | complex vertical displacement | squared Hankel-beam displacement |
| Noise FK/FJ | GPa / Pa, reference spectral forcing | expected covariance | FK expected power; FJ **magnitude** of signed transform |

For coherent arrays, retained direct `response` is a spatial spectral displacement: length² for FK and length³ for FJ. `array_response` and `receiver_response` are length. PSD is applied to image power, not to these coherent complex amplitudes. Noise covariances include PSD. Metadata records the units, including the separate `direct_power_units`.

For `exp(i k x+i ωt)`, coherent FK first synthesizes
`u(r)=∫[G(k)exp(i k r)+G(−k)exp(−i k r)] dk/(2π)`.
The Rayleigh negative branch is obtained from physical parity `C(−k)=D C(k) D`, `D=diag(−1,1)`; Love is even. The steered stack is `Σ w(r) u(r) exp(i k_out r)/Σw`. This is a **2-D line-force** experiment; it must not be compared as an absolute-amplitude 3-D point-source experiment.

Active FJ synthesizes `u_z(r)=∫ J0(kr) G_zz(k) k dk/(2π)` and applies `∫ w(r)r J0(k_out r)u_z(r)dr / ∫w(r)r dr`. It currently requires a **vertical point source and vertical Rayleigh receivers**. Radial here means source–receiver distance, not the radial displacement component. Horizontal point-force/SH FJ, VTI and buried sources are not implemented. The complex `receiver_response[...,0]` stores those vertical radial records. Finite apertures, discrete distances and windows broaden peaks and produce sidelobes; this is not an additional determinant/root calculation.

Noise FJ uses an explicitly defined **isotropic, band-limited, uncorrelated spatial Fourier forcing ensemble**, with source polarization expressed in the rotating radial/vertical or transverse basis. Its pair covariance is
`C(r)=∫J0(kr) P(k) k dk / ∫k dk`, `P=source_psd*|G_selected|²` (or the vector trace).
It then applies the same area-normalized radial J0 transform. `array_response` preserves the **signed** FJ transform; `power=abs(array_response)` has covariance units, not covariance squared. The ensemble normalization fixes total forcing power over the specified band. Changing the band changes the source ensemble as well as resolution. This is not a universal ambient-noise Green-function retrieval law, an `Im(G)`/equipartition prescription, or a localized collection of uncorrelated real-space point sources. No causal phase is discarded from coherent records to make a noise image.

For noise, vertical Rayleigh means ZZ covariance; radial Rayleigh and Love selections mean the **horizontal vector trace**, and `total` means the corresponding vector trace. They are not pair-oriented TT/RR projections. Station positions may be `[N,2]` for FJ. Equivalent pair distances are merged, and radial integration uses trapezoidal distance quadrature, **not pair-count weighting**. `station_pair_count` retains the original count. `radii=` may directly supply unique observed pair distances. Noise FK instead defines one-way incoherent plane waves along the line and uses the squared array response; it is not the 2-D isotropic-noise operator.

Geometry, frequency/velocity meshes and damping are fixed inputs. Cached Fourier/Bessel bases and subsequent model-dependent transforms stay on the selected CPU/CUDA device; only fixed-grid cache keys and validation scalars are inspected on the host. `max_cache_mb` guards an estimated basis/workspace size, not the entire solver/AD graph. The CPU array operator needs the optional PyTorch dependency even for NumPy inputs. Geometry uses model length units. A line may be set using explicit positions or two consistent values among `length`, `n_receivers`, `spacing`; FK accepts only a line. Windows are rectangular, Hann or Hamming. At least two sampled distances with nonzero window weight are required.

The supplied reference slide motivated the side-by-side comparison layout, not a quantitative validation against its unavailable input data or SPECFEM2D/SASEM/VTI runs. Relevant observation-method references: [array response and modal interference](https://academic.oup.com/gji/article/218/1/295/5420372), [FJ, geometry and isotropic-source assumptions](https://academic.oup.com/gji/article/237/1/250/7589741).

## Analytic image gradients and inversion

```python
import torch
model = torch.tensor(layers, dtype=torch.float64, device='cpu', requires_grad=True)
operator = SurfaceWave(model)                 # use CUDA tensors for CUDA
prediction = operator.dispersion_image(f, c, array=active, normalization='none')
# observed must use the same source, units, processing and normalization.
observed = torch.zeros_like(prediction['power'])  # replace with data
valid = prediction['valid'] & torch.isfinite(observed)
loss = .5 * (prediction['power'][valid] - observed[valid]).square().sum()
loss.backward()                                # model.grad: same shape/units

# Convenience interface, including NumPy models, masks and data weights:
grad = sw.image_sensitivity(f, c, observed=observed.detach().cpu().numpy(),
                           field='power', array=active, normalization='none')
# grad['model_gradient'], ['objective'], ['prediction'], ['valid']
# Or supply cotangent=... for Re(sum(conj(cotangent)*prediction)).
```

The physical adjoint solves `Tᴴ λ = Uᴴ g`, then contracts `dU` and `dT` with amplitudes/adjoint fields. It propagates the Hermitian adjoint through bounded interface/reflection recurrences, contracting local analytic eigenmatrix derivatives for Vp, Vs, density and layer thickness. One upward state pass and one downward adjoint produce all layer gradients; no finite-difference model perturbations and no full pixel-by-parameter Jacobian. Work and state grow O(number of layers), so cost is **not independent of layer count**. CUDA processes models and frequency/velocity nodes together and reduces gradients on the same stream. Its adjoint state/partial-gradient workspace is chunked to 128 MiB; this does not bound all image/array memory. CPU reduces per-worker gradients. Array, power and normalization derivatives use Torch's analytic reverse operations.

The native boundary adjoint supports **first derivatives only**. Torch models, complex source amplitudes and nonnegative PSD inputs may be trainable. `image_sensitivity` returns source/PSD gradients when numeric values are supplied, and a model gradient in the original depth/thickness parameterization and units. Top depth zero and unused half-space thickness have zero derivatives; optional Q columns have zero derivatives in this elastic model. Validity changes and modal poles are not differentiated. Invalid data/prediction pixels are excluded from objectives; an array frequency row is invalid if any integration sample fails. Missing roots are never filled into training labels. For higher-order derivatives, fluids, anisotropy or material attenuation, a different supported physical formulation is needed.
