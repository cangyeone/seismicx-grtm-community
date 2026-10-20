# Complete-method binary preview — 0.6.0.dev3

This preview integrates GRTM Green functions, structural derivatives, RF and
HMC/DRAM interfaces, and adds Rayleigh/Love dispersion with semi-analytic kernels.
Complete implementation source remains in the maintainer repository. This
community repository distributes compiled packages, API documentation and
numerical results under the existing SeismicX GRTM research/noncommercial license.

## Download and install

Download a matching wheel from the [release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev3).
CPython 3.10–3.14 wheels cover Apple Silicon macOS 11+ (CPU + Metal) and
Linux/WSL x86_64, glibc 2.28+ (CPU + CUDA). Check the
[SHA-256 manifest](../downloads/SHA256SUMS-0.6.0.dev3) before installing the exact file.

```sh
python -m pip install /path/to/downloaded.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

Stable PyPI remains 0.5.1; ordinary `pip install seismicx-grtm` does not obtain
these preview features. NumPy CPU surface-wave calls do not need PyTorch.
CUDA and RF/HMC require a compatible PyTorch distribution. CUDA binaries target
compute capability 12.0, tested on RTX 5090 with Linux/Python 3.13/Torch 2.8.0+cu128.
Other Python versions also require a PyTorch distribution supporting that Python,
GPU and driver. macOS surface waves currently use CPU; Metal in the package is
for point-source Green functions.

## Dispersion and kernels from a velocity model

```python
import numpy as np
from grtm import SurfaceWave

# Top depth km, Vp/Vs km/s, density g/cm3; final layer is a halfspace.
layers = np.array([[0., 3.2, 1.65, 2.1], [.8, 5.6, 3.15, 2.6],
                   [5., 6.7, 3.8, 2.9], [25., 8.1, 4.6, 3.3]])
sw = SurfaceWave(layers, device='cpu')  # 'cuda:0' on compatible Linux/CUDA
periods = [5., 10., 20.]
c = sw.phase_velocity(periods, wave='rayleigh')
u = sw.group_velocity(periods, wave='rayleigh')
k = sw.sensitivity(periods, kind='group', parameter='velocity_s')['kernel']
print(c, u, k.shape)
```

The [surface-wave guide](SURFACE_WAVES.md) covers semi-analytic mathematics,
batches, units, parameter conventions, autograd and scope. The
[timing report](SURFACE_WAVE_BENCHMARKS.md) separates pure CPU, CUDA dispersion and
complete kernels: CPU Vs kernels are about 1.81× / 1.59× faster than disba;
a separate CUDA campaign measured 27.4× / 41.1× against disba CPU. Dispersion
alone currently does not outperform disba's default CPU forward solver.

This release also exposes `JointRFForward`, `ShallowCrustModel`, `JointRFProblem`,
`hmc_sample`, `dram_sample` and diagnostics. The [CPU RF guide](DIFFERENTIABLE_RF_CPU.md)
describes the historical 0.6.0.dev2 preview; the integrated RF API in this version
supports CPU and native CUDA. See the [release verification](../verification/preview-0.6.0.dev3.json)
for binary audits and actual device-test scope.
