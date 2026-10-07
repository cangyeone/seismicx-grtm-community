# Installation and backend selection

English | [简体中文](INSTALLATION.zh-CN.md) | [Home](../README.md)

## Install from PyPI

Use a virtual environment with CPython 3.10–3.14:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install seismicx-grtm==0.5.1
python -m pip check
python -m grtm --info
```

The `source` command above applies to macOS/Linux/WSL shells. Native Windows
wheels are not supplied; install within Linux under WSL. `seismicx-grtm` is the
distribution name; **`grtm` is the import name**. Do not install an unrelated
package named `grtm` to obtain SeismicX GRTM.

| Platform | Wheel tag | Included engines |
|---|---|---|
| Apple Silicon, macOS 11+ | `macosx_11_0_arm64` | CPU, Metal |
| Linux/WSL x86_64, glibc 2.28+ | `manylinux_2_28_x86_64` | CPU, CUDA |

Each Python minor version has its own wheel. For example `cp312-cp312` means
CPython 3.12; it cannot be installed into CPython 3.11. PyPy, free-threaded
CPython builds, Intel macOS, Linux ARM and native Windows are not covered.
There is no source-build fallback.

## Optional dependencies

```sh
python -m pip install 'seismicx-grtm[rf]==0.5.1'
python -m pip install 'seismicx-grtm[torch]==0.5.1'
python -m pip install 'seismicx-grtm[jax]==0.5.1'
```

NumPy is the core dependency. `rf` adds SciPy for inversion; synthesis and
H–κ stacking need only NumPy. The framework extras install compatible framework
packages, but GPU-enabled framework builds have their own driver/runtime
requirements. A PyTorch/JAX GPU tensor device and the GRTM solver backend are
separate settings. Importing `grtm` alone does not import either framework.

## Choose an engine

```python
import grtm
print(grtm.available_backends())
print(grtm.build_info())
solver = grtm.Solver(backend="cpu")
# Linux, supported NVIDIA GPU:
# solver = grtm.Solver(backend="cuda")
# Apple Silicon:
# solver = grtm.Solver(backend="metal")  # aliases: "mps", "apple"
```

CPU is the default; `"c"` is its alias. `available_backends()` reports libraries
included in the wheel, not whether the machine can execute each one. Backend
failures raise errors instead of silently switching engines.

The Linux wheel includes CUDA 12.8 runtime support, statically linked. You do not
need `nvcc` or a local CUDA toolkit. **CUDA execution requires an NVIDIA driver
compatible with CUDA 12.8 and compute capability 12.0.** RTX 5090 is the tested
GPU. This wheel has no CUDA kernels for older GPU architectures; use CPU there.
The CPU engine works without an NVIDIA GPU or driver. In WSL, the Windows host
driver must provide GPU access to WSL.

The Apple wheel contains a precompiled Metal library, so neither Xcode nor a
shader compiler is required at runtime. M4 Max is the physically tested device;
other Apple Silicon models have not been separately validated. `mps` names the
custom hybrid Metal engine, not an all-PyTorch implementation.

## Download a wheel or install offline

Use the [download matrix](../downloads/README.md). Files on GitHub Releases and
PyPI are byte-identical. Check the file against [SHA256SUMS](../downloads/SHA256SUMS).

```sh
python -m pip install /path/to/downloaded-wheel.whl
```

For offline use, on an online machine matching the target Python and platform:

```sh
python -m pip download --only-binary=:all: --dest wheelhouse 'seismicx-grtm[rf]==0.5.1'
# Transfer wheelhouse to the offline machine, then:
python -m pip install --no-index --find-links=wheelhouse 'seismicx-grtm[rf]==0.5.1'
```

NumPy and optional dependencies must also be available offline. A single GRTM
wheel is not a complete offline environment.

## Upgrade and verify

```sh
python -m pip install --upgrade seismicx-grtm
python -c "import grtm; print(grtm.__version__); print(grtm.available_backends())"
```

If migrating from the historical `grtm-green` distribution, uninstall that
distribution first because it owns the same `grtm` import directory. Use one
environment per application when dependency requirements differ.

See [troubleshooting](TROUBLESHOOTING.md) if pip cannot find a matching wheel.
