# Phase/group velocity and oscillatory quadrature preview — 0.6.0.dev6

[Release / 二进制下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev6)

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev6-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

Replace the placeholder tags with the actual matching filename. CPython 3.10–3.14; Apple Silicon macOS 11+ (CPU/Metal), Linux/WSL x86_64 glibc≥2.28 (CPU/CUDA). CUDA targets compute capability 12.0, verified with RTX 5090 and Torch 2.8.0+cu128. No compiler is needed to install. Framework adapters and CUDA dispersion additionally require compatible Torch.

Accelerates CUDA phase/group velocity roots without relaxing FP64, mode checks or root tolerances. Adds CPU batched Levin, joint field/analytic-JVP integration and optional `method="auto"` Gauss/Levin selection. Adaptive wavenumber quadrature is CPU-only; surface-wave CUDA forward/derivatives stay on GPU. Other device boundaries match dev5: Green FFT remains on host, and MPS RF/surface is an explicit CPU FP64 bridge.

[Surface-wave report](PERFORMANCE_0.6.0.dev6.md) · [Quadrature / 积分报告](QUADRATURE_0.6.0.dev6.md) · [Device boundaries](DEVICE_EXECUTION.md)

See reports for precision scopes and retained failures. No universal infinite-tail certificate or blanket matched-error speed advantage over PTAM is claimed. The research/noncommercial licence is unchanged; community downloads contain no solver source. GitHub preview only; stable PyPI is unchanged.
