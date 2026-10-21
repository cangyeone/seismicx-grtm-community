# Device execution and derivatives binary preview — 0.6.0.dev5

English | [简体中文](PREVIEW_0.6.0.dev5.zh-CN.md)

Download a [matching wheel](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev5)
and install its actual filename (the tags below are placeholders):

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev5-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

Wheels cover CPython 3.10–3.14, Apple Silicon macOS 11+ (CPU/Metal), and
Linux/WSL x86_64 glibc≥2.28 (CPU/CUDA). CUDA targets compute capability 12.0,
tested on RTX 5090. No compiler/nvcc is needed to install a wheel. Framework
adapters, RF and CUDA surface waves additionally need a Torch distribution
compatible with Python/GPU; this RTX 5090 verification uses Torch 2.8.0+cu128.

Adds shared native CPU RF forward/VJP, CUDA analytic Green structural
integrals/contractions, batched backward, reusable CUDA buffers and reduced RF
VJP temporaries. Semi-analytic derivatives do not perturb the model; surface
FP64 and the default 1e-10 km/s tolerance are unchanged. RF/surface forward and
derivatives stay on CUDA. Green low-frequency extended precision and FFT retain
host boundaries. MPS RF/surface is an explicit CPU FP64 bridge, not native GPU
acceleration.

See [simple APIs/device boundaries](DEVICE_EXECUTION.md) and
[timing/accuracy evidence](PERFORMANCE_0.6.0.dev5.md). CPU is the default. CUDA
errors do not silently fall back. Custom backward supports first-order only.
Green CUDA VJP currently integrates selected parameter directions; it is not
the CPU reverse-AD algorithm and is not guaranteed faster.

The research/noncommercial licence is unchanged. Community downloads contain
no solver implementation source. Stable PyPI remains 0.5.1; this preview was not
uploaded to PyPI. SHA-256, wheel manifests and verification accompany the assets.
