# 0.6.0.dev11 binary preview

Download from the [public prerelease](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev11). Install the wheel matching Python and platform:

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev11-*.whl
```

- CPython 3.12, Apple Silicon macOS 11+: CPU + Metal package. Surface images compute on CPU FP64; MPS is an explicit CPU bridge.
- CPython 3.13, manylinux_2_31 x86_64: CPU + CUDA 12.8, compute capability 12.0, tested on RTX 5090. CUDA calls additionally need compatible PyTorch and NVIDIA drivers.

New: [forced-response dispersion images](DISPERSION_IMAGE.md), returning complex displacement and raw/normalized power from surface vertical, radial or transverse loads. CPU/CUDA and fused model batches. [Accuracy and timing](PERFORMANCE_0.6.0.dev11.md).

The image API is forward-only; existing dispersion derivatives, Green/RF interfaces and dev10 CUDA Green acceleration are retained. Elastic solid layers only; no buried sources or material-Q image attenuation. The research/noncommercial licence is unchanged. This GitHub preview does not change stable PyPI. Public artifacts contain compiled implementations, documentation and numeric evidence; solver implementation source is not distributed.
