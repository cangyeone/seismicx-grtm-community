# Binary downloads

The files below are the **same ten wheels published on PyPI**, with identical
SHA-256 hashes. Select the Python version and operating system matching your
interpreter. The simple installation path is `python -m pip install seismicx-grtm`.
See [installation](../docs/INSTALLATION.md) for driver and platform requirements.

| Python | Platform and engines | Download |
|---|---|---|
| 3.10 | Apple Silicon (CPU + Metal) | [seismicx_grtm-0.5.1-cp310-cp310-macosx_11_0_arm64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp310-cp310-macosx_11_0_arm64.whl) |
| 3.10 | Linux/WSL x86_64 (CPU + CUDA) | [seismicx_grtm-0.5.1-cp310-cp310-manylinux_2_28_x86_64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | Apple Silicon (CPU + Metal) | [seismicx_grtm-0.5.1-cp311-cp311-macosx_11_0_arm64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp311-cp311-macosx_11_0_arm64.whl) |
| 3.11 | Linux/WSL x86_64 (CPU + CUDA) | [seismicx_grtm-0.5.1-cp311-cp311-manylinux_2_28_x86_64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | Apple Silicon (CPU + Metal) | [seismicx_grtm-0.5.1-cp312-cp312-macosx_11_0_arm64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp312-cp312-macosx_11_0_arm64.whl) |
| 3.12 | Linux/WSL x86_64 (CPU + CUDA) | [seismicx_grtm-0.5.1-cp312-cp312-manylinux_2_28_x86_64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | Apple Silicon (CPU + Metal) | [seismicx_grtm-0.5.1-cp313-cp313-macosx_11_0_arm64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp313-cp313-macosx_11_0_arm64.whl) |
| 3.13 | Linux/WSL x86_64 (CPU + CUDA) | [seismicx_grtm-0.5.1-cp313-cp313-manylinux_2_28_x86_64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | Apple Silicon (CPU + Metal) | [seismicx_grtm-0.5.1-cp314-cp314-macosx_11_0_arm64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp314-cp314-macosx_11_0_arm64.whl) |
| 3.14 | Linux/WSL x86_64 (CPU + CUDA) | [seismicx_grtm-0.5.1-cp314-cp314-manylinux_2_28_x86_64.whl](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.5.1/seismicx_grtm-0.5.1-cp314-cp314-manylinux_2_28_x86_64.whl) |

After downloading the matching wheel:

```sh
python -m pip install /path/to/seismicx_grtm-0.5.1-<matching-tags>.whl
```

The angle-bracket expression is a placeholder; use the actual downloaded name.
To verify a download, compare `shasum -a 256 <wheel>` (macOS) or
`sha256sum <wheel>` (Linux) with [SHA256SUMS](SHA256SUMS).
NumPy is required; an offline installation must also provide compatible NumPy
and any optional dependencies.

The GitHub release's automatic **Source code** ZIP/TAR links contain only this
community repository's documentation and data. The solver binaries are the
explicit `.whl` assets. No solver implementation source is included here.
