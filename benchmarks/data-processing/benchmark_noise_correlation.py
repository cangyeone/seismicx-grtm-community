"""Public-API CPU/CUDA timings, including transfers and numerical comparisons."""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import grtm
from grtm import noise_cross_correlation


def timed(fn, repeats, cuda=False):
    for _ in range(2):
        fn()
    if cuda:
        torch.cuda.synchronize()
        # Reach steady GPU clocks after lengthy CPU baselines, as well as
        # warming FFT plans/allocator. Preserve every timed sample below.
        warm_start = time.perf_counter()
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
        while time.perf_counter()-warm_start < .25:
            fn()
            torch.cuda.synchronize()
    measurements = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        measurements.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(measurements), samples_ms=measurements)


def errors(value, reference):
    delta = value.astype(np.float64)-reference.astype(np.float64)
    return dict(max_abs=float(np.max(abs(delta))),
                relative_l2=float(np.linalg.norm(delta.ravel())/np.linalg.norm(reference.ravel())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Use a CUDA host for this benchmark'
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    report = dict(version=grtm.__version__, python=platform.python_version(),
                  numpy=np.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cpu=cpu, logical_cpus=os.cpu_count(), gpu=torch.cuda.get_device_name(0),
                  repeats=args.repeats, cpu_warmups=2, cuda_warmup_min_seconds=.25,
                  measurement='warm synchronized wall time; no compilation/file IO; prevalidated input',
                  gpu_transfer_measurement='CPU waveform -> CUDA -> CCF -> CPU (metadata stays on device)',
                  settings=dict(dt=.02, detrend='constant', taper_fraction=.05,
                                whitening='smooth', smooth_bins=11, freqmin=.2, freqmax=10.,
                                taper_hz=.1, normalization='coeff', stack='mean', max_lag=5.,
                                check_finite=False), rows=[])
    cases = [('small', 1, 4, 2048, 512),
             ('array', 1, 32, 65536, 4096),
             ('batch4', 4, 32, 65536, 4096),
             ('dense128', 1, 128, 32768, 4096)]
    rng = np.random.default_rng(20261110)
    with torch.inference_mode():
        for name, batch, stations, samples, window in cases:
            for dtype in [np.float32, np.float64]:
                x = rng.normal(size=(batch, stations, samples)).astype(dtype)
                opts = dict(report['settings'], window_samples=window, step_samples=window//2)
                row = dict(case=name, batch=batch, stations=stations, record_samples=samples,
                           window_samples=window, windows=(samples-window)//(window//2)+1,
                           pairs=stations*(stations-1)//2, dtype=np.dtype(dtype).name)
                torch.set_num_threads(1)
                reference = noise_cross_correlation(x, **opts)['correlation']
                row['numpy_cpu'] = timed(lambda: noise_cross_correlation(x, **opts), args.repeats)
                host = torch.from_numpy(x)
                for threads in sorted(set([1, args.threads])):
                    torch.set_num_threads(threads)
                    call = lambda: noise_cross_correlation(host, **opts)
                    row[f'torch_cpu_{threads}'] = timed(call, args.repeats)
                    row[f'torch_cpu_{threads}']['error_vs_numpy'] = errors(call()['correlation'].numpy(), reference)
                gpu = host.to('cuda')
                call = lambda: noise_cross_correlation(gpu, **opts)
                torch.cuda.reset_peak_memory_stats()
                row['cuda_resident'] = timed(call, args.repeats, cuda=True)
                row['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
                actual = call()['correlation'].cpu().numpy()
                row['error_cuda_vs_numpy'] = errors(actual, reference)
                limit = 2e-5 if dtype == np.float32 else 2e-11
                assert row['error_cuda_vs_numpy']['relative_l2'] < limit, row
                transfer = lambda: noise_cross_correlation(host.to('cuda'), **opts)['correlation'].cpu()
                row['cuda_with_transfers'] = timed(transfer, args.repeats, cuda=True)
                cpu = min(row[k]['median_ms'] for k in row if k.startswith('torch_cpu_') or k == 'numpy_cpu')
                row['speedup_vs_fastest_measured_cpu'] = cpu/row['cuda_resident']['median_ms']
                row['speedup_with_transfers_vs_fastest_cpu'] = cpu/row['cuda_with_transfers']['median_ms']
                report['rows'].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(name, row['dtype'], 'CUDA ms', round(row['cuda_resident']['median_ms'], 3),
                      'speedup', round(row['speedup_vs_fastest_measured_cpu'], 2), flush=True)


if __name__ == '__main__':
    main()
