"""Installed-wheel batch latency, throughput and CPU/CUDA scaling.

RF/surface use native model-batched CUDA calls. Green functions use an explicitly
labelled host thread pool of independent model calls; structural AD is CPU work.
No forward implementation, root tolerance or numerical precision is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import gc
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import grtm
import numpy as np
import torch
from benchmark_forward_derivatives import solid, dispersion_model, array, relative, sync


def gpu_state():
    path = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    try:
        return subprocess.check_output([path, '--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total',
                                        '--format=csv,noheader'], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def measure(cases, repeat, device):
    """Rotate backends and operations; thread setup is outside the timed call."""
    samples = {name: {op: [] for op in case['functions']} for name, case in cases.items()}
    outputs = {name: {} for name in cases}
    names = list(cases)
    for name, case in cases.items():
        torch.set_num_threads(case['torch_threads'])
        for operation, function in case['functions'].items():
            outputs[name][operation] = function()
            sync(device)
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    for trial in range(repeat):
        for name in names[trial % len(names):]+names[:trial % len(names)]:
            case = cases[name]
            torch.set_num_threads(case['torch_threads'])
            operations = list(case['functions'])
            for operation in operations[trial % len(operations):]+operations[:trial % len(operations)]:
                sync(device)
                start = time.perf_counter()
                result = case['functions'][operation]()
                sync(device)
                elapsed = time.perf_counter()-start
                outputs[name][operation] = result
                samples[name][operation].append(elapsed)
                print(json.dumps(dict(progress=name, operation=operation, trial=trial+1, seconds=elapsed)), flush=True)
    peak = torch.cuda.max_memory_allocated(device)
    return samples, outputs, peak


def emit_cases(args, emit, family, batch, kind, cases, **metadata):
    before = gpu_state()
    samples, outputs, peak = measure(cases, args.repeat, args.device)
    reference = outputs[next(iter(cases))]
    for name, case in cases.items():
        accuracy = {op+'_relative_cpu_error': relative(value, reference[op])
                    for op, value in outputs[name].items()}
        if any(e > (1e-6 if op.startswith('forward_derivative') else 1e-8)
               for op, e in accuracy.items()):
            raise AssertionError((family, kind, batch, name, accuracy))
        medians = {k: statistics.median(v) for k, v in samples[name].items()}
        emit(dict(family=family, kind=kind, layers=args.layers, batch=batch, backend=name,
                  cuda_device=args.device if case['cuda'] else None,
                  forward_backend='cuda' if case['cuda'] else 'cpu',
                  derivative_backend='cpu' if family == 'green' else ('cuda' if case['cuda'] else 'cpu'),
                  cpu_threads=case['cpu_threads'], torch_threads=case['torch_threads'],
                  cuda_model_fusion=family != 'green' and case['cuda'],
                  trials_seconds=samples[name], median_seconds=medians,
                  models_per_second={k: batch/v for k, v in medians.items()},
                  milliseconds_per_model={k: 1000*v/batch for k, v in medians.items()},
                  accuracy=accuracy, gpu_state_before=before, gpu_state_after=gpu_state(),
                  torch_peak_allocated_bytes_for_whole_case=peak, **metadata))


def run_rf(args, emit):
    maximum = max(args.batches)
    models = np.repeat(solid(args.layers)[None, :, :4]/1000, maximum, 0)
    models[:, :, 2] *= np.linspace(.99, 1.01, maximum)[:, None]
    rng = np.random.default_rng(20261021)
    cot_rf = rng.normal(size=(maximum, 1, 6, 81))
    cot_vs = rng.normal(size=(maximum, 1, 6, 41))
    for batch in args.batches:
        m = models[:batch].copy()
        cases = {}
        for device, threads in [('cpu', t) for t in args.rf_threads]+[(args.device, 1)]:
            torch.set_num_threads(threads)
            op = grtm.JointRFForward([.06], device=device)
            x = torch.tensor(m, dtype=torch.float64, device=device, requires_grad=True)
            cr = torch.tensor(cot_rf[:batch]/(batch*6*81), dtype=x.dtype, device=device)
            cv = torch.tensor(cot_vs[:batch]/(batch*6*41), dtype=x.dtype, device=device)

            def forward(op=op, x=x):
                with torch.no_grad():
                    return op(x)['rf']

            def derivative(op=op, x=x, cr=cr, cv=cv):
                y = op(x)
                return torch.autograd.grad((y['rf']*cr).sum()+(y['vs_app']*cv).sum(), x)[0]

            name = 'cuda' if device != 'cpu' else 'cpu'+str(threads)
            cases[name] = dict(functions=dict(forward=forward, forward_derivative=derivative),
                               cuda=device != 'cpu', cpu_threads=threads, torch_threads=threads)
        emit_cases(args, emit, 'receiver_function', batch, None, cases,
                   model_input='resident tensor', sampling=dict(nfft=2048, samples=81, gaussians=6, rays=1, polarization_periods=41),
                   derivative_task='scalar VJP for top/Vp/Vs/rho; fixed RF/apparent-Vs cotangents; first top fixed')
        del cases
        gc.collect()
        torch.cuda.empty_cache()


def run_surface(args, emit):
    models = dispersion_model(args.layers, max(args.batches))
    periods = np.geomspace(2, 50, 12)
    for batch in args.batches:
        m = models[:batch].copy()
        for kind in ['phase', 'group']:
            cases = {}
            for device, threads in [('cpu', t) for t in args.surface_threads]+[(args.device, 1)]:
                values = torch.tensor(m, dtype=torch.float64, device=device) if device != 'cpu' else m
                op = grtm.SurfaceWave(values, parameterization='thickness', threads=threads)
                name = 'cuda' if device != 'cpu' else 'cpu'+str(threads)
                cases[name] = dict(functions={
                    'forward': lambda op=op, kind=kind: op.dispersion(periods, kind=kind)['velocity'],
                    'forward_derivative': lambda op=op, kind=kind: op.sensitivity(periods, kind=kind, parameter='velocity_s')['kernel']},
                    cuda=device != 'cpu', cpu_threads=threads, torch_threads=1)
            emit_cases(args, emit, 'dispersion', batch, kind, cases, model_input='resident model, cached search mesh',
                       periods=periods.tolist(), root_tolerance_km_s=1e-10,
                       derivative_task='all-layer Vs kernel, including roots', wave='rayleigh', mode=0)
            del cases
            gc.collect()
            torch.cuda.empty_cache()


def run_green(args, emit):
    maximum = max(args.green_batches)
    models = np.repeat(solid(args.layers)[None], maximum, 0)
    models[:, :, 2] *= np.linspace(.99, 1.01, maximum)[:, None]
    model = grtm.example_model(units='si')
    model.update(layers=solid(args.layers), source_depth=8000., log2_samples=6, dt=.4,
                 distances=np.linspace(10000, 50000, 4), t0=np.zeros(4), ptam=0)
    moment = grtm.strike_dip_rake(20, 40, 60, 1e15)
    grid = grtm.Solver().operator(model, threads=args.green_threads).freeze_grid()
    cot = np.random.default_rng(20261021).normal(size=(4, 64, 3))
    solvers = {'cpu': grtm.Solver(), 'cuda': grtm.Solver(backend='cuda')}
    with ThreadPoolExecutor(max_workers=args.green_workers) as pool:
        for batch in args.green_batches:
            ms = models[:batch]
            cases = {}
            for backend in ['cpu', 'cuda']:
                # One active model for B1; otherwise keep the total CPU budget fixed.
                workers = min(batch, args.green_workers)
                threads = max(1, args.green_threads//workers)
                options = dict(threads=threads)
                if backend == 'cuda':
                    options['device'] = torch.device(args.device).index or 0
                op = solvers[backend].operator(model, parameters=['vp', 'vs'], vjp_method='adjoint', grid=grid, **options)

                def single_forward(m, op=op):
                    return op(m, moment)

                def single_derivative(m, op=op):
                    lin = op.linearize(m, moment)
                    return lin.vjp(cot, moment_gradient=False)[0]

                def batch_forward(single=single_forward, ms=ms, workers=workers):
                    if workers == 1:
                        return np.stack([single(m) for m in ms])
                    return np.stack(list(pool.map(single, ms)))

                def batch_derivative(single=single_derivative, ms=ms, workers=workers):
                    if workers == 1:
                        return np.stack([single(m) for m in ms])
                    return np.stack(list(pool.map(single, ms)))

                functions = {'forward': batch_forward}
                if batch in args.green_gradient_batches:
                    functions['forward_derivative'] = batch_derivative
                cases[backend] = dict(functions=functions, cuda=backend == 'cuda',
                                      cpu_threads=args.green_threads, torch_threads=1)
            emit_cases(args, emit, 'green', batch, None, cases,
                       host_model_workers=min(batch, args.green_workers),
                       native_threads_per_worker=max(1, args.green_threads//min(batch, args.green_workers)),
                       batch_execution='host thread pool of separate model calls; no fused CUDA model axis',
                       model_input='host NumPy, native input/output and transfer included',
                       derivative_task='Vp/Vs scalar VJP on CPU, including base forward',
                       field='displacement', samples=64, receivers=4,
                       fixed_grid=True, wavenumber_nodes=int(np.sum(grid['steps']+1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--layers', type=int, default=32)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 16, 64, 256])
    p.add_argument('--families', nargs='+', choices=['rf', 'surface', 'green'], default=['rf', 'surface', 'green'])
    p.add_argument('--rf-threads', type=int, nargs='+', default=[1, 4])
    p.add_argument('--surface-threads', type=int, nargs='+', default=[16])
    p.add_argument('--green-batches', type=int, nargs='+', default=[1, 16, 64])
    p.add_argument('--green-gradient-batches', type=int, nargs='+', default=[1, 16])
    p.add_argument('--green-workers', type=int, default=4)
    p.add_argument('--green-threads', type=int, default=16)
    p.add_argument('--repeat', type=int, default=3)
    args = p.parse_args()
    if min(args.batches+args.green_batches+args.rf_threads+args.surface_threads+
           [args.layers, args.repeat, args.green_workers, args.green_threads]) < 1:
        p.error('positive sizes, repetitions and thread counts required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    try:
        cpu = subprocess.check_output(['lscpu'], text=True)
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    report = dict(version=grtm.__version__, installed_package=grtm.__file__,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=platform.node(), platform=platform.platform(), cpu=cpu,
                  torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(args.device),
                  gpu_state_start=gpu_state(), arguments=vars(args)|{'output': str(args.output)},
                  precision='FP64/complex128; unchanged root tolerance 1e-10 km/s',
                  protocol='Fixed model cohort, prefix batches. One warmup per operation/backend; rotating backends and operation order; synchronized host wall time. Setup excluded. RF/surface use resident model inputs, public checks included. Green includes host I/O/transfer and uses a host model thread pool, not fused CUDA models. CPU budgets are stated. Peak Torch allocation excludes native CUDA malloc. No unrelated jobs stopped.',
                  records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        report['records'].append(record)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)

    for family in args.families:
        {'rf': run_rf, 'surface': run_surface, 'green': run_green}[family](args, emit)
    report['gpu_state_end'] = gpu_state()
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
