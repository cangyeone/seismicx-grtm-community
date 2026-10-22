# 0.6.0.dev12 binary preview

Download the matching wheel from the [public prerelease](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev12):

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev12-*.whl
```

- CPython 3.12, Apple Silicon macOS 11+: CPU + Metal package. Dispersion images and their adjoints use CPU FP64; MPS explicitly bridges to CPU.
- CPython 3.13, manylinux_2_31 x86_64: CPU + CUDA 12.8, architecture 120, tested on RTX 5090. CUDA additionally requires compatible PyTorch/NVIDIA drivers.
- PyTorch is required for gradients and finite-array processing, including CPU NumPy array calls. Direct NumPy forward needs NumPy only.

New [image adjoints and finite-array observations](DISPERSION_IMAGE.md): direct spectra, coherent line-source FK, vertical point-source FJ, and separately defined isotropic-noise pair-covariance FJ. Model, complex source and source-PSD first derivatives; full model batches; same-device CUDA physics and array transforms. `image_sensitivity` provides a single-objective VJP/least-squares gradient without materializing a dense image Jacobian. [Accuracy and timing](PERFORMANCE_0.6.0.dev12.md).

Only elastic isotropic solid layers and surface sources/receivers are supported. No VTI, material-Q attenuation, fluids, buried sources, higher derivatives, or universal ambient-noise source model. All spatial synthesis is band limited; its sampling and endpoints require convergence checks. The user-supplied slide is a visual reference, not an independently reproduced SPECFEM/SASEM validation.

The research/noncommercial licence is unchanged. Public artifacts contain compiled implementations, documentation and numerical evidence; no solver implementation source. This GitHub preview does not change stable PyPI.
