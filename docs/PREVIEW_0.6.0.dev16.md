# 0.6.0.dev16: complete CPU/CUDA noise correlations

[中文](PREVIEW_0.6.0.dev16.zh-CN.md) · [GitHub wheels](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev16) · [PyPI](https://pypi.org/project/seismicx-grtm/0.6.0.dev16/)

Adds `grtm.noise_cross_correlation`: overlapping windows, detrend, time normalization,
frequency filtering/whitening, zero-padded linear CCFs, weighted mean/sum or phase-weighted
stacking, pair-local gap masks and tensor gradients. Station FFTs are shared across pairs;
linear stacks reduce cross spectra before the final inverse FFT.

```sh
python -m pip install --only-binary=:all: seismicx-grtm==0.6.0.dev16
```

This preview supplies **CPython 3.13** wheels for Apple Silicon macOS 11+ (native CPU/Metal)
and Linux/WSL x86_64, glibc 2.31+ (native CPU/CUDA 12.8 SM120). The native solver CUDA engine
requires a compatible driver and compute capability 12.0; RTX 5090 is tested. Noise processing
uses NumPy CPU or the installed PyTorch CPU/CUDA backend. PyTorch is optional for NumPy input;
install an appropriate PyTorch build for tensor input. Noise processing MPS/half precision and
other Python/platform wheel ABIs are not supported by this preview.

[Complete API and numerical conventions](NOISE_CORRELATION.md) · [Example](../examples/noise_cross_correlation.py) · [CPU/CUDA performance](PERFORMANCE_0.6.0.dev16.md) · [Verification](../verification/preview-0.6.0.dev16/verification.json) · [Checksums](../downloads/SHA256SUMS-0.6.0.dev16)

All 35 implementation modules are compiled extensions. Three minimal public package/CLI
bootstrap Python files remain; no implementation sources, headers or sdist are distributed.
The research/noncommercial license is unchanged. Mac host libraries were rebuilt; the unchanged
Metal shader is reused from the verified dev12 binary, with source/binary hashes checked.
Whitening/correlation remain on CPU/CUDA even though the package also contains the native Metal
Green-function engine. Stable PyPI 0.5.1 does not contain this API; explicitly pin dev16.
