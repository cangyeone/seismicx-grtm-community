"""Compare smooth and strict whitening of a batch of colored seismic records."""
import argparse

import numpy as np

from grtm.processing import spectral_whiten, whiten_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("numpy", "cpu", "cuda"), default="numpy")
    args = parser.parse_args()
    dt, n = 0.05, 4096
    frequency = np.fft.rfftfreq(n, dt)
    envelope = 0.1 + np.exp(-((frequency-0.7)/0.2)**2)
    rng = np.random.default_rng(20261106)
    x = np.fft.irfft(np.fft.rfft(rng.normal(size=(4, n))) * envelope, n=n)
    if args.device == "numpy":
        fft = np.fft.rfft
        to_numpy = np.asarray
    else:
        import torch
        x = torch.tensor(x, dtype=torch.float64, device=args.device, requires_grad=True)
        fft = torch.fft.rfft
        to_numpy = lambda value: value.detach().cpu().numpy()
    settings = dict(freqmin=0.2, freqmax=2., taper_hz=0.1)
    for method in ("smooth", "strict"):
        y = spectral_whiten(x, dt, method=method, smooth_bins=21, **settings)
        spectrum = whiten_spectrum(fft(x), dt, n, method=method, smooth_bins=21, **settings)
        np.testing.assert_allclose(to_numpy(fft(y)), to_numpy(spectrum), rtol=1e-12, atol=1e-12)
        selected = (frequency >= settings["freqmin"]) & (frequency <= settings["freqmax"])
        amplitude = abs(to_numpy(spectrum)[..., selected])
        if method == "strict":
            np.testing.assert_allclose(amplitude, 1., rtol=1e-12)
        print(f"{method}: shape={tuple(y.shape)}, dtype={y.dtype}, "
              f"passband amplitude range=[{amplitude.min():.4g}, {amplitude.max():.4g}]")
        if args.device != "numpy":
            gradient, = torch.autograd.grad(y.square().mean(), x)
            print(f"  forward device={y.device}, gradient device={gradient.device}")


if __name__ == "__main__":
    main()
