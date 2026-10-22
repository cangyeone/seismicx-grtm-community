# Explicit Green wavefield kernels — 0.6.0.dev9

[Binary release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev9) · [Methods/API](GREEN_WAVEFIELD_KERNELS.md) · [Measurements](PERFORMANCE_0.6.0.dev9.md)

| Python | Platform | Included backends |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU and Metal; the new explicit Green kernels run on CPU |
| CPython 3.13 | Linux/WSL x86_64, glibc ≥ 2.31 | CPU and CUDA sm120, tested on RTX 5090 |

Download the matching wheel and install it with `python -m pip install /path/to/wheel.whl`. Install matching PyTorch separately for tensor APIs. Other dev9 ABIs are not published; dev7 remains available for its existing matrix. Stable PyPI is unchanged. Do not rename wheel tags.

Set `vjp_method='wavefield'` on `grtm.torch.Forward` or `ForwardOperator` to select the new exact layer overlap integral; `adjoint` remains available and the default policy is unchanged. Both support FP64 CPU/CUDA. Near-static precision guards retain the old extended-precision AD path; trace coincidence, very thin sublayers or conditioning failures also retain the verified fallback. Inspect execution statistics. This release does not claim native FP64 MPS wavefield kernels or a fused cross-model Green CUDA launch.

Both wheels are built from source commit `f3d8106fc059ab75ba5580fbaa3fadcaeb908917`. Linux is repaired with auditwheel for glibc 2.31. All Mac native host libraries and Python extensions are rebuilt. Only the unchanged compiled Metal shader is extracted from a SHA256-verified dev7 reference, with its source and binary hashes checked; no shader source is embedded. This differs from dev8's reuse of a whole native Metal library. Provenance is recorded in build metadata and verification JSON.

The CUDA runtime shutdown guard fixes a reproduced autograd-worker teardown crash, and subprocess regression checks are included. The research/noncommercial licence is unchanged. Community publication contains compiled implementation, documentation and numerical evidence; solver source remains in the full repository.
