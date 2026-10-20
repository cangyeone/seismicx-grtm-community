"""CPU GRTM vs original single-thread disba: dispersion and complete Vs kernels."""
import argparse
import json
import platform
import time
from pathlib import Path

import disba
from disba import PhaseDispersion, GroupDispersion, PhaseSensitivity, GroupSensitivity
import numpy as np
import grtm
from grtm import SurfaceWave


def model(n):
    h = np.r_[np.linspace(.5, 1.8, n - 1), 0.]
    vs = np.linspace(1.9, 4.4, n)
    vp = 1.8 * vs
    rho = np.linspace(2.2, 3.2, n)
    vs[-1], vp[-1], rho[-1] = 4.6, 8.1, 3.3
    return np.stack((h, vp, vs, rho), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers', nargs='+', type=int, default=[8, 32])
    p.add_argument('--batches', nargs='+', type=int, default=[1, 16])
    p.add_argument('--periods', type=int, default=12)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--threads',type=int,default=1,help='Native GRTM model workers; disba stays unchanged/single-threaded')
    p.add_argument('--tasks', nargs='+', choices=['dispersion', 'sensitivity'],
                   default=['dispersion', 'sensitivity'])
    p.add_argument('--wave', choices=['rayleigh', 'love'], default='rayleigh')
    p.add_argument('--output', type=Path, default=Path('surface-cpu-benchmark.json'))
    a = p.parse_args()
    periods = np.geomspace(2., 50., a.periods)
    records = []
    report = dict(host=platform.node(), cpu=platform.processor(), disba=disba.__version__,grtm=grtm.__version__,
                  precision='FP64/complex128', periods=periods.tolist(), records=records,grtm_threads=a.threads,
                  protocol='Pure CPU; original disba single-threaded, GRTM requested native model workers '+str(a.threads)+'. Warmed complete '
                  'batches, median of repeated host wall times. Prepared operators, construction '
                  'excluded. No CUDA calls, no model autograd. disba Dunkin, dc=.005, dp=.025, '
                  'group dt=.025. GRTM analytic derivatives and search=auto, root tolerance 1e-10 km/s. '
                  'Shared host, no other users jobs stopped.')
    for n in a.layers:
        for batch in a.batches:
            m = np.repeat(model(n)[None], batch, axis=0)
            m[:, :, 2] *= np.linspace(.99, 1.01, batch)[:, None]
            for task in a.tasks:
                for kind in ('phase', 'group'):
                    cls = ({'phase': PhaseDispersion, 'group': GroupDispersion} if task == 'dispersion'
                           else {'phase': PhaseSensitivity, 'group': GroupSensitivity})[kind]
                    refs = [cls(*mi.T, algorithm='dunkin') for mi in m]
                    op = SurfaceWave(m, parameterization='thickness',threads=a.threads)

                    def baseline():
                        if task == 'dispersion':
                            return np.stack([ref(periods, wave=a.wave).velocity for ref in refs])
                        return np.stack([[ref(float(t), wave=a.wave, parameter='velocity_s').kernel
                                          for t in periods] for ref in refs])

                    def target():
                        if task == 'dispersion':
                            return op.dispersion(periods, wave=a.wave, kind=kind)['velocity']
                        return op.sensitivity(periods, wave=a.wave, kind=kind,
                                              parameter='velocity_s')['kernel']

                    baseline()
                    target()
                    old, new = [], []
                    for _ in range(a.repeat):
                        start = time.perf_counter()
                        expected = baseline()
                        old.append(time.perf_counter() - start)
                        start = time.perf_counter()
                        actual = target()
                        new.append(time.perf_counter() - start)
                    r = dict(layers=n, batch=batch, period_count=len(periods), task=task, kind=kind,
                             wave=a.wave, disba_seconds=old, grtm_cpu_seconds=new,
                             disba_median=float(np.median(old)), grtm_cpu_median=float(np.median(new)),
                             speedup=float(np.median(old) / np.median(new)),
                             max_absolute_difference=float(np.abs(actual - expected).max()),
                             difference_note='Group disba has a frequency-difference approximation; '
                             'its kernels also have a 2.5% one-sided model perturbation. '
                             'These are not derivative-accuracy bounds.')
                    records.append(r)
                    a.output.parent.mkdir(parents=True, exist_ok=True)
                    a.output.write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
