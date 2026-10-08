# Release 0.5.1 verification

All ten CPython 3.10–3.14 wheels passed package-content, license, checksum and
metadata checks, plus CPU interface/numerical regression tests. Every wheel had
25 passing tests and 9 explicit GPU/optional-framework skips in its build environment.

| Installed binary | Device tests | Scope |
|---|---|---|
| macOS arm64, Python 3.14 | 32 passed, 2 skipped | Apple M4 Max, CPU/Metal, PyTorch MPS and JAX callbacks; CUDA skipped |
| Linux x86_64, Python 3.13 | 29 passed, 5 skipped | RTX 5090 native CUDA, CPU, CPU PyTorch/JAX adapters; Metal skipped |

After publication, the package was reinstalled from PyPI on both machines.
CPU/Metal and CPU/CUDA calculations passed smoke comparisons. Those small-model
errors are diagnostic examples, not accuracy guarantees for arbitrary models.
All ten PyPI wheel hashes were checked against the validated files; no source
archive was uploaded. The GitHub Release assets use these same files.

- [Executed documentation examples](documentation-examples.json): forward, batch, Jacobian/VJP, CLI, RF inversion, PyTorch/JAX, and Metal/MPS on the installed 0.5.1 package.
- [Release summary](release-0.5.1.json)
- [Mac public-install smoke result](macos-public-smoke.json)
- [Linux public-install smoke result](linux-public-smoke.json)
- [Binary checksums](../downloads/SHA256SUMS)
- [Scientific limits](../docs/NUMERICAL_NOTES.md)

Only two plain Python files remain inside a wheel: public imports and the CLI
entry point. The eleven implementation modules are compiled extensions. Native
implementation sources and headers are omitted; Metal shaders are precompiled.
This prevents direct delivery of implementation source, but does not make
machine code impossible to reverse engineer.

## Research preview verification

[0.6.0.dev1](preview-0.6.0.dev1.md) has its own binary-content audits and
installed-package tests. The counts above remain the historical 0.5.1 results.
