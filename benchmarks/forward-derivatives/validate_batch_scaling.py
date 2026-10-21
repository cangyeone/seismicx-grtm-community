"""Check native batching against individual calls and CPU adjoint scheduling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import grtm
import numpy as np
import torch
from benchmark_forward_derivatives import solid, dispersion_model, array, relative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(a.device)
    results = []
    n = 32
    models = np.repeat(solid(n)[None, :, :4]/1000, 256, 0)
    models[:, :, 2] *= np.linspace(.99, 1.01, 256)[:, None]
    x = torch.tensor(models[:16], dtype=torch.float64, device=a.device, requires_grad=True)
    rf = grtm.JointRFForward([.06], device=a.device)
    y = rf(x)
    g = torch.autograd.grad(y['rf'].square().sum()+y['vs_app'].square().sum(), x)[0]
    for i in [0, 8, 15]:
        xi = x[i:i+1].detach().requires_grad_()
        yi = rf(xi)
        gi = torch.autograd.grad(yi['rf'].square().sum()+yi['vs_app'].square().sum(), xi)[0]
        e = dict(family='receiver_function', batch=16, model=i,
                 forward_relative_error=relative(y['rf'][i:i+1], yi['rf']),
                 vjp_relative_error=relative(g[i:i+1], gi))
        assert max(e['forward_relative_error'], e['vjp_relative_error']) < 1e-10, e
        results.append(e)

    m = dispersion_model(n, 256)
    periods = np.geomspace(2, 50, 12)
    sw = grtm.SurfaceWave(m, device=a.device, parameterization='thickness')
    for kind in ['phase', 'group']:
        batched = sw.sensitivity(periods, kind=kind, parameter='velocity_s')
        for i in [0, 128, 255]:
            one = grtm.SurfaceWave(m[i], device=a.device, parameterization='thickness').sensitivity(periods, kind=kind, parameter='velocity_s')
            e = dict(family='dispersion', kind=kind, batch=256, model=i,
                     forward_relative_error=relative(batched['velocity'][i], one['velocity']),
                     kernel_relative_error=relative(batched['kernel'][i], one['kernel']))
            assert max(e['forward_relative_error'], e['kernel_relative_error']) < 1e-7, e
            results.append(e)

    models = np.repeat(solid(n)[None], 64, 0)
    models[:, :, 2] *= np.linspace(.99, 1.01, 64)[:, None]
    selected = models[[0, 8, 15]]
    model = grtm.example_model(units='si')
    model.update(layers=solid(n), source_depth=8000., log2_samples=6, dt=.4,
                 distances=np.linspace(10000, 50000, 4), t0=np.zeros(4), ptam=0)
    solver = grtm.Solver()
    grid = solver.operator(model, threads=16).freeze_grid()
    serial = solver.operator(model, parameters=['vp', 'vs'], vjp_method='adjoint', grid=grid, threads=16)
    concurrent = solver.operator(model, parameters=['vp', 'vs'], vjp_method='adjoint', grid=grid, threads=4)
    mt = grtm.strike_dip_rake(20, 40, 60, 1e15)
    cot = np.random.default_rng(20261021).normal(size=(4, 64, 3))

    def call(op, model):
        return op.linearize(model, mt).vjp(cot, moment_gradient=False)[0]

    reference = np.stack([call(serial, mi) for mi in selected])
    with ThreadPoolExecutor(max_workers=4) as pool:
        batch = np.stack(list(pool.map(lambda mi: call(concurrent, mi), selected)))
    e = dict(family='green', models=3, serial_threads=16, model_workers=4,
             native_threads_per_worker=4, derivative_backend='cpu',
             vjp_relative_error=relative(batch, reference))
    assert e['vjp_relative_error'] < 1e-10, e
    results.append(e)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(version=grtm.__version__, device=a.device, precision='FP64/complex128',
                  scope='RF batch16 versus separate GPU calls; surface batch256 thread selector versus single-model warp selector; Green concurrent versus serial CPU adjoints. Representative models, not a universal bound.',
                  records=results)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
