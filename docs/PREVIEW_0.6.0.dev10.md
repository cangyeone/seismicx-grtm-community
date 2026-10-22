# Fused Earth-model CUDA Green functions — 0.6.0.dev10

[Binary release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev10) · [API and precision policy](CUDA_GREEN_BATCH.md) · [Measurements](PERFORMANCE_0.6.0.dev10.md)

| Python | Platform | Backends |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU and Metal; explicit Green derivatives use CPU |
| CPython 3.13 | Linux/WSL x86_64, glibc ≥2.31 | CPU and CUDA sm120, tested on RTX 5090 |

Download the matching wheel and install using `python -m pip install /path/to/wheel.whl`. Install matching Torch separately. Do not rename wheel tags. Only these two ABIs are published for dev10; stable PyPI is unchanged.

CUDA `grtm.torch.Forward(..., vjp_method="wavefield")` fuses Earth batches by default. Default `cuda_precision="strict"` preserves the original precision guards. Explicit `cuda_precision="fast"` keeps eligible low-frequency propagation and explicit gradients on GPU. Extreme-frequency models retain CPU protection; special geometry can retain reverse AD. Validation and topology planning remain host work. Inspect `execution_stats()`; fused counts cover the whole batch.

Both wheels' solver and Python extensions are built from source commit `a130cc274d70239130a25f0c0bf8936682378c54`. Linux is auditwheel-repaired for glibc 2.31. All Mac host libraries and Python extensions are rebuilt; only the unchanged compiled dev7 Metal shader is reused, with reference wheel, shader source and binary hashes recorded. Shader source is not embedded.

CPU, MPS, RF and dispersion algorithms have no new changes or speedup claims. Surface-wave documentation adds missing-mode masks, shared train/inference preprocessing, fixed normalization, model-family validation and chunked storage conventions. No training dataset or network is published.

The research/noncommercial licence remains unchanged. Full solver sources are in the source repository; community publishes compiled implementation, documentation and numerical verification only. Hashes, logs and packaging audits are in the [verification directory](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev10).
