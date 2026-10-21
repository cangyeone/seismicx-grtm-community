# Analytic adjoints and CUDA reverse AD preview — 0.6.0.dev7

[Release / 二进制下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev7)

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev7-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

Replace placeholder tags with the actual matching filename. CPython 3.10–3.14; Apple Silicon macOS 11+ (CPU/Metal), Linux/WSL x86_64 glibc≥2.28 (CPU/CUDA). CUDA targets compute capability 12.0, tested on RTX 5090 with Torch 2.8.0+cu128. Installation needs no compiler; tensor interfaces and CUDA dispersion also require compatible Torch.

GRTM characteristic adjoints and analytic mixed derivatives compute phase/group kernels. RF uses native material/matrix adjoints. Green adds true CUDA reverse AD and retains Torch spectra, physical synthesis, FFT and source gradients on GPU. Ill-conditioned low-frequency bins retain host extended precision. CPU Levin can return physical waveforms and analytic JVPs jointly; RF FFT and dispersion roots are not labelled as Levin.

[Methods / 公式与接口](ANALYTIC_METHODS.md) · [Performance / 实测](PERFORMANCE_0.6.0.dev7.md) · [Devices](DEVICE_EXECUTION.md)

Not every small workload is faster; the report includes regressions. Adaptive Levin is CPU-only; MPS RF/surface remains a CPU FP64 bridge. The research/noncommercial licence is unchanged and solver implementation source is not distributed here. GitHub preview only; stable PyPI is unchanged.
