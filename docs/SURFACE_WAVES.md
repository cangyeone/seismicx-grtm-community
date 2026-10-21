# GRTM surface waves and semi-analytic sensitivity kernels

English | [简体中文](SURFACE_WAVES.zh-CN.md)

`SurfaceWave` extends this project's GRTM eigenmatrices, interface continuity and decaying half-space conditions. Native CPU and CUDA implementations share FP64/complex128 mathematics. Production calculations do not import disba or perturb/re-solve the model once per parameter.

## Build and use

This feature is available in the
[0.6.0.dev4 binary preview](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev4),
not in stable PyPI 0.5.1. Download the wheel matching your Python/platform and
install that exact file with `python -m pip install /path/to/downloaded.whl`.
CUDA calls require PyTorch (`python -m pip install 'torch>=2.4'`) and a compatible
NVIDIA driver. Linux preview wheels target compute capability 12.0, tested on
RTX 5090. macOS surface-wave calls currently use CPU; Metal support in the
package applies to point-source Green functions.

Maintainers with implementation-source access can build instead:

```sh
python -m pip install '.[surface]' \
  -Ccmake.define.GRTM_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120
```

Architecture 120 is the tested RTX 5090 target; select the appropriate CUDA architecture for another GPU. CPU builds use `pip install .` and do not require Torch for NumPy calls. CUDA requires Torch and the native library, and never silently falls back to CPU.

```python
import numpy as np
from grtm import SurfaceWave, surface_wave_kernels

# GRTM columns: top depth, Vp, Vs, density; final layer is a half-space.
layers = np.array([[0,3.2,1.65,2.1], [.8,5.6,3.15,2.6],
                   [5,6.7,3.8,2.9], [25,8.1,4.6,3.3]])
periods = np.geomspace(2,40,16)
sw = SurfaceWave(layers, device="cuda:0", units="native")
c = sw.phase_velocity(periods, wave="rayleigh")
result = sw.sensitivity(periods, wave="rayleigh", kind="phase",
                        parameter="velocity_s")
print(result["kernel"].shape)  # [period, layer]
group = sw.sensitivity(periods, wave="love", kind="group", parameter="all")
```

Run `python examples/surface_wave_cuda.py --device cuda:0` for a complete example, including model batches and autograd. The one-call helpers are `surface_wave_dispersion` and `surface_wave_kernels`.

## Model and result conventions

Models are `[L,4/6]` or `[B,L,4/6]`, also accepted through a GRTM dictionary's `layers` entry. Default top depths start at zero and increase strictly. Optional Q columns are ignored by this explicitly elastic operator. Set `parameterization="thickness"` for `[h,Vp,Vs,rho]`, or call `SurfaceWave.from_thickness(h,vp,vs,rho,...)`; final half-space thickness is ignored.

Native units are km, km/s and g/cm3. `units="si"` selects m, m/s and kg/m3. Periods are always seconds. Velocities and depths follow input units. Solid, isotropic, horizontal elastic layers require positive density/Vs and Vp>sqrt(4/3)Vs. Fluids, attenuation and leaky modes are not supported.

Velocity shapes are `[P]` or `[B,P]`, kernels `[P,L]` or `[B,P,L]`; scalar periods retain the period axis. `parameter="all"` adds a last axis in the order thickness/Vp/Vs/density; a tuple/list selects an explicit order. A single parameter name removes that last axis. NumPy CPU input produces NumPy output; CUDA and tensor input produce tensors. Results include velocity, phase_velocity, period, kernel, depth, valid and normalized secular residuals.

`thickness` varies a finite layer independently, moving deeper interfaces; `top` instead moves a GRTM interface. `top[0]` is fixed and internal depth derivatives equal adjacent thickness-kernel differences. Half-space thickness and Love Vp kernels are exactly zero. Material properties are independent; empirical parameter relations can be applied through an external tensor mapping. `normalized=True` returns dimensionless logarithmic kernels.
These are discrete layer-parameter kernels. A finite-layer velocity kernel divided by layer thickness is a layer-averaged per-depth kernel; this conversion is not defined for the half-space.

## GRTM mathematics and differentiation

The source is `mat_PSV_E` in `grtm_all/grtm_c/src/popm/func/eigmatrix.c` and the continuity, propagation and half-space conditions in `grtc.c`. The new shared `src/surface_wave.hpp` uses the same P–SV eigenmatrix and SH traction relation. The parity of GRTM's matrix reduces an interface map to two 2×2 solves.

A surface wave is a homogeneous free-surface solution, or a pole of the layered Green function. Direct real-axis surface-impedance determinants have displacement poles. The same GRTM layer maps are therefore propagated in normalized decaying-subspace coordinates: six exterior-square minors for Rayleigh waves and displacement/traction for Love waves. Real exponential growth is factored out at each layer. This is not a port of the disba/CPS Dunkin or fast-delta recurrence. The original point-source, RF, C/CUDA/Metal operators retain their existing implementations.

For a simple isolated secular root F(c,T,m)=0:

```
c_m = -F_m/F_c
c_T = -F_T/F_c
U   = c*c / (c + T*c_T)
c_Tm = -(F_Tm + F_Tc*c_m)/F_c + F_T*(F_cm + F_cc*c_m)/(F_c*F_c)
U_m  = (2*c*c_m*(c+T*c_T) - c*c*(c_m+T*c_Tm)) / (c+T*c_T)**2
```

Dual and hyper-dual arithmetic differentiate the eigenmatrices, continuity solves and propagation analytically. Mixed nilpotent coefficients give the second secular derivatives needed by group kernels. There is no model-perturbation step size and no frequency-differenced approximation of group velocity. Central differences appear only in audits.

CUDA parallelizes models, periods, search points and parameter tasks on the caller's device/stream. This version propagates an analytic tangent per parameter; it avoids repeated model root searches, but does not claim linear layer-count complexity for the complete Jacobian. Small search-grid planning/sorting occurs on the host; secular evaluation, root refinement and derivatives remain on the selected backend.

Tensor velocities from `phase_velocity`, `group_velocity` or `dispersion` support first-order model autograd, including GRTM top-depth conversion and external control-point mappings. Repeated forward/backward calls are supported. A second differentiation through the native custom backward is not supported. Explicit `sensitivity()` outputs are derivative data without a second-derivative autograd graph. Periods are fixed inputs.

## Modal discovery and limitations

Mode 0 is fundamental; higher modes are roots ordered by phase speed at each input period. Period order is preserved. Searches stop below half-space Vs. A homogeneous solid half-space has a Rayleigh mode and no Love mode. Missing/unresolved modes raise by default; `missing="nan"` retains periods and a validity mask.

The default base spacing `dc` corresponds to 0.005 km/s (5 m/s for SI). Per-layer vertical-slowness grids additionally resolve modes accumulating next to material speeds in thick/short-period layers. `dc` is a search setting, not a differentiation increment. Root tolerance defaults to 1e-10 km/s, with an additional normalized residual check. `max_grid_points=20001` bounds allocations and can be increased explicitly.

Halve dc to inspect close higher modes and check continuity under model changes. Near cutoff, root multiplicity or exactly critical modal bases, ordinary derivatives may be ill-conditioned or undefined. Exactly critical bases use a tiny deterministic limiting displacement. Finite sampling is not a proof that all modes have been counted. A small residual alone does not guarantee correct modal identity.

## Reproduction

```sh
python -m pip install disba==0.7.0 pytest  # validation/benchmark only
python -m pytest tests/test_surface_wave.py -q
python examples/validate_surface_wave.py --device cuda:0 --output validation.json
python benchmarks/surface-wave/benchmark_surface_wave.py --device cuda:0 --output benchmark.json
```

Checks cover analytic half-space Rayleigh speed, an independent two-layer Love equation, a thick-layer/short-period missed-mode case, independent re-solved-model FD audits of every phase/group property, density scaling, layer-thickness/period scaling, SI/native units, batches, top-depth derivatives, both GPU devices, non-default streams and autograd.

An initial 16-model six-layer cohort (half with low-velocity layers) has maximum speed disagreement about 2.44e-6 km/s across 394 fundamental/first-overtone comparisons with disba 0.7.0's default Dunkin solver. A separate full-property FD audit has maximum absolute kernel disagreement about 2.51e-8. These are observed test errors, not universal error bounds.

The benchmark measures complete Vs-kernel computation, including root search, derivatives and synchronization, against warmed disba CPU calls. Disba's default 2.5% one-sided parameter perturbation is an approximation, not derivative ground truth. Setup/JIT and steady calls are separated; single-model latency, batches, phase and group kernels are reported separately. Local evidence is in `optimization/surface-wave-20261020/`.

### Final CMake-library measurements

RTX 5090 and Threadripper PRO 3955WX; 32 layers, 12 periods, 16 models, Rayleigh Vs kernels. Synchronized whole-batch host times are medians of two warmed repetitions. Setup/model transfer is recorded separately. These were shared devices, with no other jobs stopped.

| CPU baseline | Kernel | CPU s | CUDA s | Ratio |
|---|---|---:|---:|---:|
| Default disba FD | Phase | 6.804 | 0.248 | 27.4× |
| Default disba FD | Group | 13.646 | 0.332 | 41.1× |
| Identical native GRTM semi-analytic math, one thread | Phase | 3.969 | 0.248 | 16.0× |
| Identical native GRTM semi-analytic math, one thread | Group | 9.153 | 0.332 | 27.6× |

The disba comparison combines algorithm and device differences; the second baseline fixes the mathematics. Raw repetitions are in `final-benchmark.json` and `final-same-math-benchmark.json`. The initial 8/32-layer, 1/16/64-model sweep is retained in `benchmark.json`; batch ratios must not be substituted for single-model latency.

For the fixed seeded audit model, maximum absolute derivative discrepancies against independent re-solved-model central differences are:

| Wave / kernel | GRTM semi-analytic | Default disba FD |
|---|---:|---:|
| Rayleigh / phase | 2.51e-8 | 2.81e-2 |
| Rayleigh / group | 1.86e-8 | 3.31e-2 |
| Love / phase | 8.39e-9 | 2.91e-2 |
| Love / group | 1.27e-8 | 6.32e-3 |

These compare derivative approximations for this one audit model in native units, not geological reconstruction accuracy or a universal bound. Disba uses its default 2.5% one-sided parameter perturbation and, for group kernels, its default frequency difference. Full evidence is in `cuda-validation.json`.

See the [0.6.0.dev4 CPU/CUDA timing and accuracy report](PERFORMANCE_0.6.0.dev4.md).
The [0.6.0.dev3 measurements](SURFACE_WAVE_BENCHMARKS.md) are historical results before the search optimization.

## Optimized search (0.6.0.dev4)

See the [B1/16/64/256 scaling sweep](BATCH_SCALING_20261021.md) and
[model-batch examples and timing definitions](BATCH_EXECUTION.md).
`SurfaceWave` has native CUDA model batching; its throughput ratios differ
from both single-call latency and acceleration against CPU model workers.

`search="auto"` derives a five-component real trial operator from the same GRTM eigenmatrix. Safeguarded interpolation is followed by an original complex sign bracket and residual check. The requested root tolerance is unchanged. A Rayleigh variational lower bound skips a root-free range. CPU continuation is restricted to moderate, normally graded profiles and dense periods; LVZs, strong jumps, higher modes and sparse periods retain full discovery. `search="scan"` retains the original complex full-scan reference.

Group velocities use analytic log-c/log-T tangents through the real GRTM recurrence, with complex fallback for ill-conditioned states. Model kernels retain the original complex dual/hyper-dual implementation. CPU defaults to `threads=1`; `threads=16` parallelizes independent models and their kernels. CUDA work stays on the GPU.
