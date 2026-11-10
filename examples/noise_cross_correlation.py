"""Window, whiten, correlate and stack a station array on CPU or CUDA."""
import argparse
import numpy as np
from grtm import noise_cross_correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['numpy', 'cpu', 'cuda'], default='numpy')
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 8192)).astype(np.float32)
    # A known 7-sample delay, zero padded rather than circularly shifted.
    x[1] = np.r_[np.zeros(7), x[0, :-7]]
    if args.device != 'numpy':
        import torch
        x = torch.tensor(x, device=args.device)
    out = noise_cross_correlation(x, dt=.02, window_samples=2048, step_samples=1024,
        pairs=[[0, 1], [0, 2], [1, 2]], max_lag=2., whitening='smooth',
        freqmin=.2, freqmax=10., smooth_bins=11, stack='mean', normalization='coeff')
    ccf = out['correlation']
    print('CCF shape:', tuple(ccf.shape), 'device:', getattr(ccf, 'device', 'cpu'))
    k = int(ccf[0].argmax())
    print('station 1 delay:', float(out['lags'][k]), 'seconds (expected 0.14)')
    print('valid windows:', out['window_count'])


if __name__ == '__main__':
    main()
