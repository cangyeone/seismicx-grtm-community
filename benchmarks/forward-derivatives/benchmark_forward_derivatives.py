"""Measure public forward/first-derivative APIs on one CPU/CUDA host.

Run an installed wheel, not a source checkout on PYTHONPATH. CPU thread counts
are explicit. CUDA Green-function structural derivatives still execute on CPU;
the report records that hybrid boundary instead of calling it CUDA AD.
"""
import argparse
import datetime
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import grtm


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def relative(a, b):
    return float(np.linalg.norm(array(a)-array(b))/max(np.linalg.norm(array(b)), 1e-300))


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def measure(functions, device, repeat):
    """One warmup per operation, rotating trial order, synchronized wall time."""
    samples = {name: [] for name in functions}
    result = {}
    for name, function in functions.items():
        result[name] = function()
        sync(device)
    names = list(functions)
    for trial in range(repeat):
        for name in names[trial % len(names):]+names[:trial % len(names)]:
            sync(device)
            start = time.perf_counter()
            result[name] = functions[name]()
            sync(device)
            samples[name].append(time.perf_counter()-start)
            print(json.dumps(dict(progress=name,trial=trial+1,seconds=samples[name][-1])),flush=True)
    return dict(trials_seconds=samples,
                median_seconds={k: statistics.median(v) for k, v in samples.items()}), result


def solid(count):
    x = np.linspace(0, 1, count)
    return np.column_stack((45000*x, 6100+1900*x, 3500+950*x,
                            2700+600*x, np.full(count, 600), np.full(count, 300)))


def dispersion_model(count, batch):
    vs = np.linspace(1.9, 4.4, count)
    m = np.stack((np.r_[np.linspace(.5, 1.8, count-1), 0],
                  1.8*vs, vs, np.linspace(2.2, 3.2, count)), -1)
    m[-1, 1:] = [8.1, 4.6, 3.3]
    m = np.repeat(m[None], batch, 0)
    m[:, :, 2] *= np.linspace(.99, 1.01, batch)[:, None]
    return m


def green(args, emit):
    moment = grtm.strike_dip_rake(20, 40, 60, 1e15)
    for count in args.layers:
        layers = solid(count)
        model = grtm.example_model(units='si')
        model.update(layers=layers, source_depth=8000., log2_samples=6, dt=.4,
                     distances=np.linspace(10000, 50000, 4), t0=np.zeros(4), ptam=0)
        reference = None
        reference_grad = None
        grid = None
        rng = np.random.default_rng(20261020)
        direction = np.zeros_like(layers)
        direction[:, 1:3] = rng.normal(size=(count, 2))*10
        cot = rng.normal(size=(4, 64, 3))
        for backend, threads in [('cpu', t) for t in args.threads]+(
                [('cuda', t) for t in args.threads] if args.device.startswith('cuda') else []):
            options = dict(threads=threads)
            if backend == 'cuda':
                options['device'] = torch.device(args.device).index or 0
            start = time.perf_counter()
            op = grtm.Solver(backend=backend).operator(model, parameters=['vp', 'vs'],
                                                       vjp_method='adjoint', **options)
            if grid is None:
                grid = op.freeze_grid(layers)
            linear = op.linearize(layers, moment, grid=grid)
            sync(args.device if backend == 'cuda' else 'cpu')
            setup = time.perf_counter()-start

            def full_vjp():
                lin = op.linearize(layers, moment, grid=grid)
                return lin.vjp(cot, moment_gradient=False)[0]

            functions = {
                'forward': lambda: op(layers, moment, grid=grid),
                'jvp_only': lambda: linear.jvp(direction),
                'vjp_only': lambda: linear.vjp(cot, moment_gradient=False)[0],
                'forward_vjp': full_vjp,
            }
            timing, outputs = measure(functions, args.device if backend == 'cuda' else 'cpu', args.repeat)
            if reference is None:
                reference = outputs['forward'].copy()
                reference_grad = outputs['vjp_only'].copy()
            lhs = float(np.vdot(cot, outputs['jvp_only']).real)
            rhs = float(np.vdot(direction, outputs['vjp_only']).real)
            accuracy = dict(forward_relative_cpu_error=relative(outputs['forward'], reference),
                            vjp_relative_cpu_error=relative(outputs['vjp_only'], reference_grad),
                            adjoint_identity_relative_error=abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-300))
            if backend == 'cpu' and threads == args.threads[0]:
                h = .01
                fd = (op(layers+h*direction, moment, grid=grid)-op(layers-h*direction, moment, grid=grid))/(2*h)
                accuracy['jvp_central_difference_relative_error'] = relative(outputs['jvp_only'], fd)
            if accuracy['forward_relative_cpu_error'] > 1e-5 or accuracy['adjoint_identity_relative_error'] > 2e-7:
                raise AssertionError(accuracy)
            emit(dict(family='green', field='displacement', domain='signal', layers=count, batch=1, receivers=4, samples=64,
                      structural_parameters=2*count, parameter_names=['vp', 'vs'],
                      forward_backend=backend, derivative_backend='cpu', cpu_threads=threads,
                      cuda_device=args.device if backend == 'cuda' else None,
                      device_note='CUDA forward plus CPU JVP/VJP' if backend == 'cuda' else 'native CPU',
                      wavenumber_nodes=int(np.sum(grid['steps']+1)), fixed_grid=True,
                      setup_and_first_forward_seconds=setup, accuracy=accuracy, **timing))


def rf(args, emit):
    for count in args.layers:
        for batch in args.batches:
            m = np.repeat(solid(count)[None, :, :4]/1000, batch, 0)
            m[:, :, 2] *= np.linspace(.99, 1.01, batch)[:, None]
            ref = None
            ref_grad = None
            rng = np.random.default_rng(20261020)
            wrf = rng.normal(size=(batch, 1, 6, 81))/(batch*6*81)
            wvs = rng.normal(size=(batch, 1, 6, 41))/(batch*6*41)
            for device, threads in [('cpu', t) for t in args.rf_threads]+(
                    [(args.device, 1)] if args.device.startswith('cuda') else []):
                torch.set_num_threads(threads)
                start = time.perf_counter()
                op = grtm.JointRFForward([.06], device=device)
                x = torch.tensor(m, dtype=torch.float64, device=device, requires_grad=True)
                cr = torch.tensor(wrf, dtype=x.dtype, device=device)
                cv = torch.tensor(wvs, dtype=x.dtype, device=device)
                sync(device)
                setup = time.perf_counter()-start

                def forward():
                    with torch.no_grad():
                        y = op(x)
                    return y['rf'], y['vs_app']

                def objective(y):
                    return (y['rf']*cr).sum()+(y['vs_app']*cv).sum()

                def forward_vjp():
                    y = op(x)
                    return torch.autograd.grad(objective(y), x)[0]

                timing, outputs = measure({'forward': forward, 'forward_vjp': forward_vjp}, device, args.repeat)
                # A retained graph gives a separately labelled backward-only cost.
                value = objective(op(x))
                backward, _ = measure({'vjp_only': lambda: torch.autograd.grad(value, x, retain_graph=True)[0]},
                                      device, args.repeat)
                timing['trials_seconds'].update(backward['trials_seconds'])
                timing['median_seconds'].update(backward['median_seconds'])
                actual = array(outputs['forward'][0])
                grad = array(outputs['forward_vjp'])
                if ref is None:
                    ref, ref_grad = actual.copy(), grad.copy()
                accuracy = dict(rf_relative_cpu_error=relative(actual, ref),
                                rf_max_abs_cpu_error=float(np.max(np.abs(actual-ref))),
                                vjp_relative_cpu_error=relative(grad, ref_grad))
                if device == 'cpu' and threads == args.rf_threads[0]:
                    direction = rng.normal(size=m.shape)*m*.001
                    direction[:, 0, 0] = 0
                    with torch.no_grad():
                        d = torch.tensor(direction, dtype=x.dtype, device=device)
                        h = .01
                        fd = float((objective(op(x+h*d))-objective(op(x-h*d)))/(2*h))
                    ad = float(np.sum(grad*direction))
                    accuracy['directional_vjp_central_difference_relative_error'] = abs(ad-fd)/max(abs(ad), abs(fd), 1e-300)
                if accuracy['rf_relative_cpu_error'] > 1e-8 or accuracy['vjp_relative_cpu_error'] > 1e-6:
                    raise AssertionError(accuracy)
                emit(dict(family='receiver_function', layers=count, batch=batch,
                          nfft=2048, samples=81, rays=1, gaussians=6, polarization_periods=41,
                          parameter_names=['top', 'vp', 'vs', 'rho'],
                          objective='fixed real cotangent on RF and apparent Vs',
                          forward_backend='cuda' if device.startswith('cuda') else 'torch-cpu',
                          derivative_backend='cuda' if device.startswith('cuda') else 'torch-cpu',
                          cpu_threads=threads, cuda_device=device if device.startswith('cuda') else None,
                          setup_seconds=setup, accuracy=accuracy, **timing))


def surface(args, emit):
    periods = np.geomspace(2, 50, 12)
    for count in args.layers:
        for batch in args.batches:
            model = dispersion_model(count, batch)
            for kind in ['phase', 'group']:
                reference = None
                for device, threads in [('cpu', t) for t in args.threads]+(
                        [(args.device, 1)] if args.device.startswith('cuda') else []):
                    torch.set_num_threads(1)
                    start = time.perf_counter()
                    values = (torch.tensor(model, dtype=torch.float64, device=device)
                              if device.startswith('cuda') else model)
                    op = grtm.SurfaceWave(values, parameterization='thickness', threads=threads)
                    sync(device)
                    setup = time.perf_counter()-start
                    functions = {
                        'forward': lambda: op.dispersion(periods, kind=kind)['velocity'],
                        'forward_vs_kernel': lambda: op.sensitivity(periods, kind=kind, parameter='velocity_s')['kernel'],
                    }
                    timing, outputs = measure(functions, device, args.repeat)
                    if reference is None:
                        reference = {k: array(v).copy() for k, v in outputs.items()}
                    accuracy = {k+'_relative_cpu_error': relative(v, reference[k]) for k, v in outputs.items()}
                    if accuracy['forward_relative_cpu_error'] > 1e-8 or accuracy['forward_vs_kernel_relative_cpu_error'] > 1e-6:
                        raise AssertionError(accuracy)
                    emit(dict(family='dispersion', kind=kind, wave='rayleigh', mode=0,
                              layers=count, batch=batch, periods=periods.tolist(),
                              parameter_names=['velocity_s'], kernel_shape=[batch, 12, count],
                              forward_backend='cuda' if device.startswith('cuda') else 'cpu',
                              derivative_backend='cuda' if device.startswith('cuda') else 'cpu',
                              cpu_threads=threads, cuda_device=device if device.startswith('cuda') else None,
                              setup_seconds=setup, accuracy=accuracy, **timing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--rf-threads', type=int, nargs='+', default=[1, 4],
                        help='Torch CPU RF threads; small complex solves can scale poorly at 16')
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--families', nargs='+', choices=['green', 'rf', 'surface'], default=['green', 'rf', 'surface'])
    args = parser.parse_args()
    if args.repeat < 1 or any(t < 1 for t in args.threads+args.rf_threads):
        parser.error('positive repeat and thread counts required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    try:
        cpu = subprocess.check_output(['lscpu'], text=True)
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    report = dict(version=grtm.__version__, installed_package=grtm.__file__,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=platform.node(), platform=platform.platform(), cpu=cpu,
                  torch=torch.__version__, numpy=np.__version__, shared_hardware=True,
                  gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                  protocol='FP64/complex128. Prepared operators, one warmup, rotating operation order, whole-call host wall time and CUDA synchronization. Public validation included. Setup/model upload excluded and separately recorded. Green uses a fixed frequency-wavenumber grid and host NumPy I/O; RF/surface CUDA use resident tensors. vjp_only reuses the base linearization/graph, forward_vjp rebuilds it. No full dense Green/RF Jacobian is claimed.',
                  arguments=vars(args)|{'output': str(args.output)}, records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        report['records'].append(record)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)

    for family in args.families:
        {'green': green, 'rf': rf, 'surface': surface}[family](args, emit)
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
