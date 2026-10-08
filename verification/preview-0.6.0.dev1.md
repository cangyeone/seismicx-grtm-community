# Research preview 0.6.0.dev1 verification

[Preview guide](../docs/RESEARCH_PREVIEW.md) · [Checksums](../downloads/SHA256SUMS-0.6.0.dev1)

Ten CPython 3.10–3.14 wheels were built from private source commit
`712ab70cf1ee1d9289be5a3f21d818a6d394dacc`. Every wheel passed installation,
metadata, licence, RECORD and source-content audits, with 12 compiled Python
implementation modules. CPU + Metal is included on macOS arm64; CPU + CUDA
12.8 (compute capability 12.0) is included on manylinux_2_28 x86_64.

| Environment | Results | Scope |
|---|---|---|
| Each of five macOS CI wheels | 47 passed, 11 skipped | Installed-package suite; explicit optional/device skips |
| Each of five Linux CI wheels | 46 passed, 12 skipped | Installed-package suite; explicit optional/device skips |
| Apple M4 Max, CPython 3.14 | 51 passed, 7 skipped; additional Metal suite 5/5 passed | 56 unique tests passed; CPU/Metal, real MPS tensors, PyTorch/JAX and new derivatives |
| Remote RTX 5090, CPython 3.13 | 53 passed, 5 Metal-only skips | CPU/CUDA, PyTorch/JAX callbacks and new derivatives |

All four Python blocks in the English preview guide were executed successfully
against the exact candidate binaries on both machines. CPU/GPU spectral differences
in a small smoke model were approximately 2.09e-15 (Metal) and 4.70e-15 (CUDA).
These are consistency diagnostics, not general accuracy guarantees or timings.
The remote suite was rerun after supplying the legacy compatibility CLI's
private source fixtures, which are not part of the public wheel.

The only plain Python wheel files are the import and command-line entry shims.
Implementation source and headers are absent; Metal shaders are precompiled.
The public documentation allowlist excludes manuscript TeX, bibliography files,
development validation reports and unpublished experiment archives. Binary
packaging does not promise resistance to reverse engineering.

- [Machine-readable release summary](preview-0.6.0.dev1.json)
- [Mac installed-package smoke result](preview-macos-0.6.0.dev1.json)
- [Linux installed-package smoke result](preview-linux-0.6.0.dev1.json)

Native quadrature still returns `certified=False`; matrix certificates and
analytic reference enclosures have separately stated scopes. Adaptive quadrature
has no PyTorch/JAX backward adapter. Structural derivative products run on CPU.
No new FK speed campaign is claimed. This preview was uploaded only to GitHub
Releases; stable PyPI 0.5.1 remains unchanged.
