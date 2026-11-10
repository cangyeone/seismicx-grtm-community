# Spectral whitening

For complete CPU/CUDA windowed correlations and stacks, see [noise cross-correlation](NOISE_CORRELATION.md) (dev16).

English | [简体中文](DATA_PROCESSING.zh-CN.md)

The `grtm.processing` module provides **smooth spectral whitening** and
**strict (phase-only) whitening** for observed or synthetic seismic records.
Both functions are also exported from `grtm` in preview `0.6.0.dev15`.
See [installation](PREVIEW_0.6.0.dev15.md); stable PyPI 0.5.1 does not include it.

```python
from grtm.processing import spectral_whiten, whiten_spectrum

# traces: [station, sample], or any batch shape with a selected sample axis.
smooth = spectral_whiten(traces, dt=0.05, freqmin=0.1, freqmax=2.0,
                         method="smooth", smooth_bins=11, taper_hz=0.05)
strict = spectral_whiten(traces, dt=0.05, freqmin=0.1, freqmax=2.0,
                         method="strict", taper_hz=0.05)
```

## Definitions

Let `X[k]` be the unnormalized one-sided FFT, `A[k] = abs(X[k])`, and `W[k]`
the band window. All spectra use the standard NumPy/PyTorch backward FFT
normalization. Each trace is processed independently.

| Method | Formula | Accepted names |
|---|---|---|
| Smooth | `W[k] * X[k] / max(mean_window(A)[k], floor)` | `smooth`, `rma` |
| Strict | `W[k] * X[k] / A[k]` for valid bins; zero otherwise | `strict`, `brutal`, `phase_only` |

Names are case-insensitive. Smooth whitening follows the amplitude-envelope
normalization described by [Bensen et al. (2007), section 2.2](https://ciei.colorado.edu/ambient_noise/pubs/GJI_Bensen_etal2007_final.pdf).
The [NoisePy tutorial](https://noisepy.github.io/NoisePy/get_started.html) uses
`rma`; [MSNoise's correlation workflow](https://msnoise.org/doc/workflow/005_compute_cc.html)
describes the `Brutal` phase-only convention. These aliases select the
mathematical methods, not byte-for-byte compatibility with the external
packages' complete processing pipelines or smoothing/taper defaults.
The [MSNoise 1.6 configuration source](https://github.com/ROBelgium/MSNoise/blob/1.6/msnoise/default.py#L50)
also records `Brutal` (amplitude to 1.0) as its default whitening type.

`smooth_bins` is the **total odd window length**, not a half-width. The
centered arithmetic mean uses replicated endpoints and is computed over the
whole amplitude spectrum before applying the frequency window. Its nominal
width is `smooth_bins / (n_samples * dt)` Hz. It must be positive, odd and no
larger than the one-sided spectrum. It is ignored in strict mode. With width
1 and `water_level=0`, smooth whitening reduces to strict whitening.

`floor = water_level * max(A[passband])` is computed independently per trace.
The default `water_level=1e-12` prevents amplification of numerical zeros.
Smooth mode floors the divisor; strict mode keeps unit amplitude only where
`A > floor` and `A > 0`, and zeros the other bins. Use `water_level=0` for
strict unit amplitude at **every nonzero** passband bin. Exact zero bins and
all-zero records remain zero; their phase is undefined and is not invented.
Both methods preserve the phase of retained bins and remove DC.

## Frequency band, axes and units

- `dt`: sampling interval in **seconds**; `freqmin`, `freqmax`, `taper_hz`: **Hz**.
- Require `0 <= freqmin < freqmax <= 0.5/dt`, with at least one positive FFT
  bin in the passband. The default band spans DC to Nyquist, with DC removed.
- `taper_hz=0` gives a rectangular band. A positive value adds half-cosine
  shoulders **outside** `[freqmin, freqmax]`, clipped to zero and Nyquist.
  The passband stays flat; the shoulders transition from one to zero.
- `axis=-1` selects samples/frequencies. All other dimensions are batch axes.
  For `[batch, sample, component]`, use `axis=1`.
- Input arrays are not modified. Shape, real/complex dtype and odd/even record
  lengths are preserved. NumPy integer waveforms are promoted to float64.
- The waveform API uses the input length exactly, without zero padding. It
  returns real time samples. The spectrum API returns complex rFFT bins.

For an existing FFT, avoid recomputing it:

```python
import numpy as np

n = traces.shape[-1]
X = np.fft.rfft(traces, axis=-1)
Xw = whiten_spectrum(X, dt=0.05, n_samples=n,
                    freqmin=0.1, freqmax=2.0, method="strict")
whitened = np.fft.irfft(Xw, n=n, axis=-1)
```

`n_samples` is required because the spectrum length alone cannot distinguish
odd and even FFTs. Supply a valid rFFT spectrum: real DC and, for even lengths,
a real Nyquist endpoint. Handle data gaps, detrending and time-domain tapering
before whitening. There is no implicit time normalization or final RMS scaling.
Whitening removes absolute amplitude information; use identical settings when
comparing processed observations and predictions.

## CPU, CUDA and gradients

NumPy float32/64 and complex64/128 run on CPU. PyTorch tensors of those dtypes
run on their input CPU or CUDA device, including FFT, smoothing, normalization
and inverse FFT. No NumPy conversion, host array copy or automatic device
fallback occurs on the tensor path. PyTorch is optional; after installing the dev15 wheel, install a CPU/CUDA PyTorch
build appropriate for your device (`pip install torch` for the default build).
Float16/bfloat16 and MPS are currently rejected explicitly.

```python
import torch
from grtm.processing import spectral_whiten

x = torch.randn(32, 4096, dtype=torch.float64, device="cuda", requires_grad=True)
y = spectral_whiten(x, dt=0.05, freqmin=0.1, freqmax=2.0,
                    method="smooth", smooth_bins=11, check_finite=False)
gradient, = torch.autograd.grad(y.square().mean(), x)
assert y.device == x.device == gradient.device
```

The default `check_finite=True` rejects NaN/Inf, requiring **one scalar CUDA
synchronization** for validation. For previously validated tensors,
`check_finite=False` removes this check and permits an entirely device-resident
forward/backward path. Nonfinite input is unsupported when checking is disabled.
Gradients use PyTorch autograd through FFT, absolute value, averaging and
normalization; there are no production finite differences. Threshold crossings,
zero amplitudes and tied maxima have the usual piecewise-differentiable limits.
Sampling and processing settings are fixed host metadata, not trainable tensors.

Run `python examples/spectral_whitening.py --device numpy` (or `cpu`, `cuda`)
for a complete example. Formula, phase, band-edge, zero-input, batch, dtype,
gradient and device-residency tests run in the private implementation repository.
See the [public dev15 validation record](../verification/preview-0.6.0.dev15/verification.json).
