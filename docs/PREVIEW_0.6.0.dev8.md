# Eigenfunction and forced-response kernel preview — 0.6.0.dev8

[Binary release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev8) · [Methods/API](VARIATIONAL_KERNELS.md)

Two matching-ABI wheels are built, installed and tested:

| Python | Platform | Backends |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU/Metal; RF/surface MPS use the CPU FP64 bridge |
| CPython 3.13 | Linux/WSL x86_64, glibc ≥ 2.31 | CPU/CUDA, compute capability 12.0; tested on RTX 5090 |

```sh
python -m pip install /path/to/<matching-wheel>.whl
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

Install matching PyTorch separately for tensor APIs and CUDA surface waves. Other dev8 Python ABIs are not published; do not rename wheel tags. dev7 remains available for its existing ABI matrix. Stable PyPI is unchanged.

GitHub Actions jobs ended before any build steps ran, so no successful CI build is claimed. These wheels were built locally and on the CUDA host, with auditwheel repair for Linux. New CPU/RF/surface libraries and Python extensions are compiled. The unchanged Mac Metal library is reused byte-for-byte from the SHA256-verified dev7 wheel, with unchanged native sources and provenance recorded in `build-info.json`. No shader source is embedded.

Use `differentiation='eigenfunction'` for surface energy kernels and `.sensitivity` for scalar RF/Green observable gradients. The surface default remains `adjoint`. Analytical derivatives do not guarantee complete mode discovery or universally more accurate inversions. FP64 safeguards, Green quasistatic protection, MPS bridging and first-order limitations are documented in the method guide.

The research/noncommercial licence is unchanged. Community distribution contains compiled implementations, documentation and numerical evidence, not solver implementation source.
