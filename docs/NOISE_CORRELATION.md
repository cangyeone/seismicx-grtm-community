# CPU/CUDA noise cross-correlation

[中文](NOISE_CORRELATION.zh-CN.md) · [Spectral whitening](DATA_PROCESSING.md)

`grtm.noise_cross_correlation` processes real recordings shaped
`[..., station, sample]`. All leading dimensions are independent model, day or
record batches. It supports NumPy CPU and PyTorch CPU/CUDA float32/float64;
NumPy integer input is promoted to float64. MPS and half precision are rejected.
This API is introduced in **0.6.0.dev16**; dev15 only contains whitening.

```python
import torch
from grtm import noise_cross_correlation

# Load/transfer once. Keep this tensor on CUDA across processing operations.
records = torch.randn(32, 65536, device="cuda", dtype=torch.float32)
result = noise_cross_correlation(
    records, dt=0.02, window_samples=4096, step_samples=2048,
    pairs=None, max_lag=5.0,                # all i<j pairs, seconds
    detrend="constant", taper_fraction=0.05,
    whitening="smooth", smooth_bins=11, freqmin=0.2, freqmax=10.0,
    normalization="coeff", stack="mean", nan_policy="omit",
    window_batch_size=8, pair_batch_size=256,
    check_finite=False,                    # finite/valid parameters checked upstream
)
ccf, lags = result["correlation"], result["lags"]
valid = result["valid"]
assert ccf.device == lags.device == records.device
# ccf: [496 pairs, 501 lags]; missing pairs remain NaN with valid=False.
```

## Lag convention and linear correlation

For `(i,j)`, `C_ij[l] = sum_t x_i[t] * x_j[t+l]`. **Positive lag means j is
delayed relative to i.** After preprocessing, the unnormalized result agrees
with `numpy.correlate(x_j, x_i, "full")` or
[`scipy.signal.correlate(x_j, x_i, "full", method="direct")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).
Reversing a pair reverses its CCF. Auto-correlations may be supplied explicitly;
`include_auto=True` also includes them in the default upper-triangle pair list.
Explicit pairs preserve order, duplicates and reversals.

The product is `conj(FFT(x_i))*FFT(x_j)`, with padding to at least `2*N-1`.
The default `nfft` is the next power of two; a supplied value below `2*N-1` is
rejected. This is linear correlation, not a circular correlation with its ends
relabeled. `max_lag` selects symmetric integer sample lags inside the requested
seconds, including zero; default is the full `[-(N-1), N-1]` interval.
No additional `dt` is multiplied into the sample sums.

## Processing order

1. Extract complete windows, reject/mask bad station windows, apply QC weights.
2. Remove the window mean (`detrend="constant"`, default), or least-squares
   linear trend (`"linear"`); `None` preserves it.
3. Optionally apply `time_normalization="onebit"` (sign) or `"rma"`
   (divide by a centered running mean of absolute time amplitude).
   `time_smooth_samples` is an odd total sample width, with repeated endpoints.
4. Apply a half-cosine edge taper. `taper_fraction` is the fraction at **each**
   end (default 0.05, allowed 0–0.5); zero disables it.
5. Optionally filter and/or whiten each window using its original `N`-point FFT.
   `whitening="smooth"/"rma"` and `"strict"/"brutal"/"phase_only"` use the
   existing [whitening definitions](DATA_PROCESSING.md). With `whitening=None`,
   explicit `freqmin`/`freqmax` apply only the band window. With both frequencies
   omitted, frequency processing is disabled. A frequency filter removes DC.
6. Apply correlation normalization, padded FFT, pair products and stacking.

Filtering/whitening precedes the separate padded correlation FFT, so the CCF is
the linear correlation of the processed finite records. Each station FFT is
reused across all its pairs. Frequency smoothing uses the `N`-point frequency
spacing, not the padded correlation FFT spacing.

`window_samples=None` processes the whole record. `step_samples` defaults to
half the window length, must be positive and no larger than the window length.
Only complete windows are used; `window_starts` and `dropped_samples` describe
the coverage. No implicit gap interpolation, resampling, anti-alias filtering,
instrument removal, component rotation or sample-time alignment is performed.
These operations belong upstream. Identical preprocessing must be used for
observations and synthetic records. Whitening/one-bit processing remove
amplitude information; the result is not automatically a calibrated Green function.

## Normalization, weights and stacks

| `normalization` | Per-window definition |
|---|---|
| `"none"` | Raw sample sums |
| `"biased"` | Divide by `N` |
| `"unbiased"` | Divide by `N-abs(lag)`; this only corrects rectangular finite overlap, not taper/filter bias |
| `"coeff"` (default) | Divide by the full processed-window L2 norms of both traces; not a lag-local Pearson coefficient |

`window_weights` broadcasts to `[..., window, station]`. Weights must be finite
and nonnegative; zero excludes that station/window. Pair weights are the product
of its station weights. Masks and weights are applied after extracting each
window, not to overlapping raw samples globally.

| `stack` | Output and operation |
|---|---|
| `"mean"` (default) | Weighted mean, `[..., pair, lag]` |
| `"sum"` | Weighted sum, same shape |
| `None` | Separate normalized windows, `[..., window, pair, lag]`; values are not multiplied by their weights |
| `"pws"` | Weighted mean times weighted phase coherence to `pws_power` (default 2) |

Linear mean/sum stacks accumulate cross spectra before performing **one inverse
FFT per output pair**. PWS needs window CCFs: their analytic phases are computed
on the selected lag interval using a discrete Hilbert transform. Truncating that
interval changes phase estimates near its boundaries; PWS is not invariant to
`max_lag`. `pws_power=0` recovers the weighted mean. Zero analytic amplitude
contributes zero phase rather than a fabricated direction.

## Gaps, validity and outputs

`nan_policy="raise"` rejects NaN/Inf by default. `"omit"` detects nonfinite
samples on the current device and excludes each affected **station/window**.
Other stations/windows remain usable. Processed zero-energy windows are also
excluded. If no positive pair weight remains, CCF is NaN and `valid=False`.
Do not train on zero-filled invalid CCFs.

`window_count` counts contributing windows and `weight_sum` records the total
pair weight, both shaped `[..., pair]` for stacked results. For `stack=None`,
they are `[..., window, pair]`, counts are 0/1, and `weight_sum` is each pair
weight. `pairs`, `lag_samples`, `lags` (seconds) and `window_starts` (samples)
are arrays/tensors on the same device. `nfft`, window/step lengths,
`dropped_samples`, `stack`, and `lag_convention` are ordinary metadata.

## Device residency, gradients and memory

All numerical processing, gap masks, FFTs, pair products, reductions, PWS and
gradients stay on the input CPU/CUDA device. There is no CUDA-to-CPU fallback.
Lists/NumPy arrays supplied as pair indices or weights are copied once to CUDA;
tensor indices and weights must already match the record device.
Default validation can synchronize scalars for finiteness, pair bounds and
weights. `check_finite=False` skips these synchronizations for validated input;
`nan_policy="omit"` still masks gaps without a host synchronization.

`window_batch_size` and `pair_batch_size` bound forward temporaries. Dense all-pair
outputs still scale quadratically with station count; choose explicit pairs when
possible. `stack=None` retains all returned windows.
For mean/sum, persistent frequency accumulators also require
`batch * pairs * (nfft//2+1)` complex values, independent of pair block size.
Choose fewer explicit pairs if those accumulators exceed available memory.
Autograd retains tensors
needed by backward, so chunking does not bound total training-graph memory.
One-bit/sign has zero derivative almost everywhere. Masks, thresholds, zero
amplitudes and tied maxima are nondifferentiable boundaries; sampling, geometry
and options are fixed metadata. Production derivatives use PyTorch autograd,
not finite differences.

Run `python examples/noise_cross_correlation.py --device cuda` (or `cpu`/`numpy`).
Tests compare independent direct time sums, delayed impulses, reversed pairs,
weighted windows, gaps, batch/chunk invariance, preprocessing, Hilbert PWS,
NumPy/CPU/CUDA agreement, directional derivatives and nondefault CUDA streams.
The public-API benchmark script reports NumPy, one/multiple CPU threads, resident
CUDA and CUDA including waveform upload/CCF download separately; timings are
case-specific and exclude disk I/O and warm-up.
