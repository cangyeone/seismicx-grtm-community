# Binary downloads

Latest research preview: [0.6.0.dev9](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev9), with [installation/device boundaries](../docs/PREVIEW_0.6.0.dev9.md), [checksums](SHA256SUMS-0.6.0.dev9), [performance](../docs/PERFORMANCE_0.6.0.dev9.md) and [verification](../verification/preview-0.6.0.dev9/verification.json). This preview has Mac CPython 3.12 and Linux CPython 3.13 (glibc≥2.31) only; other preview ABIs remain on dev7. Stable PyPI remains 0.5.1.

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

## Research preview 0.6.0.dev1

The [GitHub prerelease](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev1)
contains separate preview wheels. Use [preview checksums](SHA256SUMS-0.6.0.dev1)
and read the [preview API guide](../docs/RESEARCH_PREVIEW.md) before upgrading.
These files are not part of the stable 0.5.1 PyPI release described above.

| Python | Apple Silicon: CPU + Metal | Linux/WSL x86_64: CPU + CUDA |
|---|---|---|
| 3.10 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp310-cp310-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp311-cp311-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp312-cp312-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp313-cp313-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp314-cp314-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev1/seismicx_grtm-0.6.0.dev1-cp314-cp314-manylinux_2_28_x86_64.whl) |

## Complete-method preview 0.6.0.dev3

[API and installation](../docs/PREVIEW_0.6.0.dev3.md) · [中文](../docs/PREVIEW_0.6.0.dev3.zh-CN.md) · [Checksums](SHA256SUMS-0.6.0.dev3)

Adds GRTM semi-analytic CPU/CUDA surface-wave kernels and integrates joint RF/HMC.
Download the exact Python/platform wheel; these files are not on stable PyPI.

| Python | Apple Silicon: CPU + Metal | Linux/WSL x86_64: CPU + CUDA |
|---|---|---|
| 3.10 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp310-cp310-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp311-cp311-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp312-cp312-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp313-cp313-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp314-cp314-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev3/seismicx_grtm-0.6.0.dev3-cp314-cp314-manylinux_2_28_x86_64.whl) |

## CPU/CUDA search and derivative preview 0.6.0.dev4

The [batch API guide](../docs/BATCH_EXECUTION.md) and
[2026-10-21 scaling sweep](../docs/BATCH_SCALING_20261021.md) use these same wheels.
No binary upgrade is needed for the new model-batch examples or benchmarks.

[Install/API](../docs/PREVIEW_0.6.0.dev4.md) · [中文](../docs/PREVIEW_0.6.0.dev4.zh-CN.md) · [CPU/CUDA report](../docs/PERFORMANCE_0.6.0.dev4.md) · [Checksums](SHA256SUMS-0.6.0.dev4)

Optimized GRTM dispersion roots and comprehensive Green/RF/dispersion derivative benchmarks. Stable PyPI remains 0.5.1.

| Python | Apple Silicon: CPU + Metal | Linux/WSL x86_64: CPU + CUDA |
|---|---|---|
| 3.10 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp310-cp310-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp311-cp311-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp312-cp312-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp313-cp313-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp314-cp314-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev4/seismicx_grtm-0.6.0.dev4-cp314-cp314-manylinux_2_28_x86_64.whl) |

## Phase/group velocity and quadrature preview 0.6.0.dev6

[Install/API](../docs/PREVIEW_0.6.0.dev6.md) · [中文](../docs/PREVIEW_0.6.0.dev6.zh-CN.md) · [Checksums](SHA256SUMS-0.6.0.dev6)

| Python | Apple Silicon: CPU + Metal | Linux/WSL x86_64: CPU + CUDA |
|---|---|---|
| 3.10 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp310-cp310-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp311-cp311-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp312-cp312-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp313-cp313-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp314-cp314-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev6/seismicx_grtm-0.6.0.dev6-cp314-cp314-manylinux_2_28_x86_64.whl) |

## Analytic adjoints preview 0.6.0.dev7

| Python | Apple Silicon: CPU + Metal | Linux/WSL x86_64: CPU + CUDA |
|---|---|---|
| 3.10 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp310-cp310-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp310-cp310-manylinux_2_28_x86_64.whl) |
| 3.11 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp311-cp311-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp311-cp311-manylinux_2_28_x86_64.whl) |
| 3.12 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp312-cp312-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp312-cp312-manylinux_2_28_x86_64.whl) |
| 3.13 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp313-cp313-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp313-cp313-manylinux_2_28_x86_64.whl) |
| 3.14 | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp314-cp314-macosx_11_0_arm64.whl) | [Download](https://github.com/cangyeone/seismicx-grtm-community/releases/download/v0.6.0.dev7/seismicx_grtm-0.6.0.dev7-cp314-cp314-manylinux_2_28_x86_64.whl) |
