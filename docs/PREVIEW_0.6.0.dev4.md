# CPU/CUDA search and derivative preview — 0.6.0.dev4

English | [简体中文](PREVIEW_0.6.0.dev4.zh-CN.md)

This preview optimizes GRTM surface-wave root discovery and group-velocity
tangents, and measures Green functions, receiver functions and dispersion with
their structural derivatives. Read the [complete timing and accuracy report](PERFORMANCE_0.6.0.dev4.md).
Stable PyPI remains 0.5.1.

The [batch API guide](BATCH_EXECUTION.md) and [2026-10-21 scaling sweep](BATCH_SCALING_20261021.md)
add per-model latency/models-per-second and explain which model batches are
native CUDA work. They use these same binaries; no version upgrade is required.

Download the matching CPython/platform wheel from the
[binary release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev4)
and install its actual filename:

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev4-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

The placeholder is not a literal filename. Wheels cover CPython 3.10–3.14,
Apple Silicon macOS 11+ (CPU/Metal), and Linux/WSL x86_64 glibc 2.28+
(CPU/CUDA). CUDA targets compute capability 12.0, tested on RTX 5090.
Torch is required for CUDA surface waves, joint RF and framework adapters;
install a Torch distribution compatible with your Python/GPU/driver. The
existing research/noncommercial licence is unchanged. Solver implementation
source is not included in the community distribution.

```python
import numpy as np
from grtm import SurfaceWave

# One model: top depth km, Vp/Vs km/s, density g/cm³.
layers = np.array([[0.,3.2,1.65,2.1], [.8,5.6,3.15,2.6],
                   [5.,6.7,3.8,2.9], [25.,8.1,4.6,3.3]])
models = np.repeat(layers[None], 16, axis=0)
sw = SurfaceWave(models, threads=16)  # explicit native CPU model workers
c = sw.phase_velocity([5.,10.,20.])
u = sw.group_velocity([5.,10.,20.])
k = sw.sensitivity([5.,10.,20.], kind='group', parameter='all')['kernel']
print(c.shape, u.shape, k.shape)

# CUDA: keep threads=1 (default). No CPU numerical fallback.
# gpu = SurfaceWave(models, device='cuda:0')
# c_gpu = gpu.phase_velocity([5.,10.,20.])

# Original complex full scan for checks, not the fast default:
reference = SurfaceWave(layers, search='scan', root_tolerance=1e-12)
```

`search="auto"` retains GRTM eigenmatrix physics, FP64/complex128 and the
1e-10 km/s default tolerance. Real trial propagation is certified using the
original complex GRTM equation. CPU fundamental-mode continuation is guarded;
low-velocity zones, large contrasts and higher modes retain full discovery.
Group forward uses analytic first derivatives. Structural phase/group kernels
retain complex dual/hyper-dual arithmetic. No disba runtime or per-parameter
model perturbation is used. See the [surface-wave guide](SURFACE_WAVES.md).

CPU `threads` defaults to 1; threading is across independent model rows.
`search="scan"` keeps the original serial root finder. CUDA uses the caller's
device/stream; mesh planning is on the host. Single small curves can be faster
on CPU. Benchmark default disba and GRTM at their stated precision settings.

Green-function structural JVP/VJP still execute on **CPU**, including CUDA-base
pipelines. Joint RF forward/backward and surface-wave kernels execute on CUDA
when selected. Metal support applies to point-source Green forward, not RF or
surface-wave CUDA APIs. All native custom backward interfaces are first-order.

The release verification and SHA-256 manifests accompany the download assets.
The performance report links raw repetitions, numerical audits and scripts
using the public APIs. The existing [forward](PYTHON_FORWARD.md),
[differentiable Green-function](RESEARCH_PREVIEW.md) and
[RF](DIFFERENTIABLE_RF_CPU.md) guides cover the other interfaces.
