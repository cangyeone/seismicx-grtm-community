"""Measure velocity-model -> dispersion only, with no sensitivity kernels.

CPU reference: warmed disba Dunkin or the identical native GRTM mathematics.
Reports prepared-operator latency and a complete call from a host model,
including construction, model transfer and search-mesh planning.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from grtm import SurfaceWave
from benchmark_surface_wave import model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--layers', type=int, nargs='+', default=[8, 32])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 16, 64])
    p.add_argument('--periods', type=int, default=12)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--wave', choices=['rayleigh', 'love'], default='rayleigh')
    p.add_argument('--kinds', nargs='+', choices=['phase', 'group'], default=['phase', 'group'])
    p.add_argument('--baseline', choices=['disba', 'grtm'], default='disba')
    p.add_argument('--output', type=Path, default=Path('surface-dispersion-benchmark.json'))
    a = p.parse_args()
    if a.baseline == 'disba':
        import disba
        from disba import PhaseDispersion, GroupDispersion
    torch.set_num_threads(1)
    periods = np.geomspace(2., 50., a.periods)
    records = []
    report = dict(gpu=torch.cuda.get_device_name(a.device), device=a.device,
                  host=platform.node(), cpu=platform.processor(), torch=torch.__version__,
                  precision='FP64/complex128', shared_hardware=True,
                  baseline=a.baseline, periods=periods.tolist(), records=records,
                  protocol='Dispersion only, no kernels or model autograd. Both implementations warmed; '
                  'sequential single-thread CPU. Synchronized host wall time for complete batches. '
                  'Prepared calls reuse model operators and search grids. Full calls start from a host '
                  'NumPy velocity model and include operator construction, transfer and grid planning. '
                  'No Numba compilation, library import or CUDA-context startup in steady timing. '
                  'Group disba uses its default dt=.025, while GRTM uses an analytic period derivative.')
    if a.baseline == 'disba':
        report['disba'] = disba.__version__
        report['disba_algorithm'] = 'dunkin, dc=.005, group dt=.025'
    for n in a.layers:
        for batch in a.batches:
            m = np.repeat(model(n)[None], batch, axis=0)
            m[:, :, 2] *= np.linspace(.99, 1.01, batch)[:, None]
            for kind in a.kinds:
                def cpu_operator():
                    if a.baseline == 'grtm':
                        return SurfaceWave(m, parameterization='thickness')
                    cls = PhaseDispersion if kind == 'phase' else GroupDispersion
                    return [cls(*mi.T, algorithm='dunkin') for mi in m]

                def cpu_call(op, check_periods=False):
                    if a.baseline == 'grtm':
                        return op.dispersion(periods, wave=a.wave, kind=kind)['velocity']
                    curves = [ref(periods, wave=a.wave) for ref in op]
                    if check_periods:
                        for curve in curves:
                            np.testing.assert_array_equal(curve.period, periods)
                    return np.stack([curve.velocity for curve in curves])

                def gpu_operator():
                    return SurfaceWave(m, device=a.device, parameterization='thickness')

                def gpu_call(op):
                    return op.dispersion(periods, wave=a.wave, kind=kind)['velocity']

                cpu = cpu_operator()
                gpu = gpu_operator()
                cpu_call(cpu, check_periods=True)
                gpu_call(gpu)
                torch.cuda.synchronize(a.device)
                timings = {key: [] for key in ('cpu_seconds', 'cuda_seconds',
                                               'cpu_full_call_seconds', 'cuda_full_call_seconds')}
                for _ in range(a.repeat):
                    t = time.perf_counter()
                    expected = cpu_call(cpu)
                    timings['cpu_seconds'].append(time.perf_counter() - t)
                    torch.cuda.synchronize(a.device)
                    t = time.perf_counter()
                    actual = gpu_call(gpu)
                    torch.cuda.synchronize(a.device)
                    timings['cuda_seconds'].append(time.perf_counter() - t)
                    t = time.perf_counter()
                    cpu_call(cpu_operator())
                    timings['cpu_full_call_seconds'].append(time.perf_counter() - t)
                    torch.cuda.synchronize(a.device)
                    t = time.perf_counter()
                    gpu_call(gpu_operator())
                    torch.cuda.synchronize(a.device)
                    timings['cuda_full_call_seconds'].append(time.perf_counter() - t)
                r = dict(layers=n, batch=batch, period_count=len(periods), kind=kind,
                         wave=a.wave, **timings)
                for key, values in timings.items():
                    r[key.replace('_seconds', '_median')] = float(np.median(values))
                r['speedup'] = r['cpu_median'] / r['cuda_median']
                r['full_call_speedup'] = r['cpu_full_call_median'] / r['cuda_full_call_median']
                r['max_velocity_difference_km_s'] = float(np.abs(actual.cpu().numpy() - expected).max())
                records.append(r)
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
