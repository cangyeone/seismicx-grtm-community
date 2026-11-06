# 0.6.0.dev15: spectral whitening binary preview

[中文](PREVIEW_0.6.0.dev15.zh-CN.md) · [Release downloads](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev15) · [Checksums](../downloads/SHA256SUMS-0.6.0.dev15) · [Verification](../verification/preview-0.6.0.dev15/verification.json)

This preview adds `grtm.processing.spectral_whiten` for waveforms and
`whiten_spectrum` for one-sided complex spectra. Both are also exported by `grtm`.
Choose smooth amplitude normalization (`smooth` / `rma`) or phase-only
normalization (`strict` / `brutal` / `phase_only`). See the
[processing guide](DATA_PROCESSING.md) and [runnable example](../examples/spectral_whitening.py).
It also includes the current array/DAS and robust surface-wave implementation.

## Install the matching binary

Download one `.whl` asset from the release and install it in a CPython 3.13 environment:

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev15-<matching-tags>.whl
python -c 'import grtm; print(grtm.__version__)'
```

Replace the placeholder with the actual filename. The two supported builds are:

| CPython | Platform | Included native engines |
|---|---|---|
| 3.13 | Apple Silicon, macOS 11+ | CPU and Metal |
| 3.13 | Linux/WSL x86_64, glibc 2.31+ | CPU and CUDA 12.8, SM120 |

CUDA solver execution requires a compatible NVIDIA driver and compute capability
12.0 (tested on RTX 5090). CPU execution does not require a GPU. This preview
does not provide other Python ABIs, Intel Mac, Linux ARM or native Windows wheels.
PyTorch is an optional dependency for tensor inputs; install a build appropriate
for the intended CPU/CUDA device. NumPy inputs only require NumPy.

The same two wheels are also [published on PyPI](https://pypi.org/project/seismicx-grtm/0.6.0.dev15/). Install the preview explicitly:

```sh
python -m pip install --only-binary=:all: seismicx-grtm==0.6.0.dev15
```

An unpinned stable install still selects 0.5.1, which does not contain whitening.

## Device and numerical boundaries

- NumPy processing runs on CPU; PyTorch float32/float64 processing and gradients
  stay on the input CPU/CUDA device. Whitening does not support MPS or half precision.
- `check_finite=True` performs one scalar validation synchronization on CUDA.
  Use `check_finite=False` only for already validated finite input to remove it.
- Smooth whitening divides by a centered running mean of spectral amplitude,
  with repeated endpoints. Strict whitening sets nonzero accepted passband bins
  to unit amplitude. Set `water_level=0` for all nonzero passband bins; zero bins
  and DC stay zero. Optional cosine shoulders lie outside the flat passband.
- These are whitening operators, not complete Bensen/NoisePy/MSNoise pipelines.
  No automatic detrending, gap filling, RMS normalization or padding is applied.

## Binary distribution and validation

All 34 implementation modules, including processing and the robust-root helpers,
are compiled extensions. Only three minimal package/CLI bootstrap `.py` files
remain. No `.pyx`, C/C++/CUDA/Metal source, headers, or source distribution is
published. Documentation, examples, model data and license notices remain readable.
Compilation does not prevent reverse engineering. The existing research/noncommercial
license is unchanged.

The verification record identifies the exact wheel hashes, package-content audits,
installed-binary tests and device/runtime versions. The Mac host libraries were
rebuilt; its unchanged precompiled Metal shader was reused only after matching the
shader-source hash and published dev12 artifact hash. Prior release performance
figures are historical; this release does not claim a new universal speedup.
