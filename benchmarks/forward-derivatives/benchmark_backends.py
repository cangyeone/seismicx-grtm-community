"""Same-host end-to-end forwards/analytic derivatives and model throughput.

CPU/CUDA/MPS arithmetic boundaries are stored with each record. Run an installed
wheel for published results. No timings from different hosts form a speed ratio.
"""
import argparse,datetime,json,platform,statistics,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import grtm


def array(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def relative(a,b):return float(np.linalg.norm(array(a)-array(b))/max(np.linalg.norm(array(b)),1e-300))
def sync(device):
    if device.startswith('cuda'):torch.cuda.synchronize(device)
    elif device=='mps':torch.mps.synchronize()
def measure(fn,device,repeat):
    fn();sync(device);ts=[]
    for _ in range(repeat):
        sync(device);t=time.perf_counter();value=fn();sync(device);ts.append(time.perf_counter()-t)
    return dict(seconds=statistics.median(ts),trials=ts),value

def layers(n,b):
    x=np.linspace(0,1,n)
    a=np.stack((45*x,6.1+1.9*x,3.5+.95*x,2.7+.6*x,np.full(n,600),np.full(n,300)),-1)
    models=np.repeat(a[None],b,0);models[:,:,2]*=np.linspace(.99,1.01,b)[:,None]
    return models

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    p.add_argument('--families',nargs='+',default=['rf','surface','green']);p.add_argument('--batches',nargs='+',type=int,default=[1,16,64,256]);p.add_argument('--layers',type=int,default=32);p.add_argument('--repeat',type=int,default=3);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--disba',action='store_true');p.add_argument('--green-layers',type=int,default=8);p.add_argument('--green-max-batch',type=int,default=16)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    records=[];report=dict(version=grtm.__version__,python=platform.python_version(),platform=platform.platform(),device=args.device,arguments=vars(args),records=records,start=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=False)
    def emit(row):
        row['models_per_second']=row['batch']/row['seconds'];records.append(row)
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    devices=['cpu']+([args.device] if args.device!='cpu' else [])
    if 'rf' in args.families:
        for b in args.batches:
            model=layers(args.layers,max(args.batches))[:b,:,:4]
            configs=[('cpu','torch',4),('cpu','native',4)]+([(args.device,'native',1 if args.device.startswith('cuda') else 4)] if args.device!='cpu' else [])
            reference={}
            for device,kernel,threads in configs:
                torch.set_num_threads(4)
                op=grtm.JointRFForward([.06],device=device,kernel=kernel,threads=threads)
                x=torch.tensor(model,dtype=torch.float32 if device=='mps' else torch.float64,device=device,requires_grad=True)
                for gradient in (False,True):
                    def run():
                        if gradient:
                            y=op(x);loss=y['rf'].square().mean()+.01*y['vs_app'].square().mean();return torch.autograd.grad(loss,x)[0]
                        with torch.no_grad():return op(x)['rf']
                    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
                    timing,value=measure(run,device,args.repeat)
                    key=gradient
                    if key not in reference:reference[key]=array(value)
                    error=relative(value,reference[key]);limit=5e-5 if device=='mps' else 2e-8
                    if error>limit:raise AssertionError(('RF mismatch',error))
                    emit(dict(family='rf',operation='forward_vjp' if gradient else 'forward',layers=args.layers,batch=b,backend=device,kernel=kernel,threads=threads,nfft=2048,error=error,compute_device=str(op.device),peak_torch_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,**timing))
    if 'surface' in args.families:
        n=args.layers;vs=np.linspace(1.9,4.4,n);m=np.stack((np.r_[np.linspace(.5,1.8,n-1),0],1.8*vs,vs,np.linspace(2.2,3.2,n)),-1);m[-1,1:]=[8.1,4.6,3.3]
        cohort=np.repeat(m[None],max(args.batches),0);cohort[:,:,2]*=np.linspace(.99,1.01,max(args.batches))[:,None];periods=np.geomspace(2,50,12)
        for b in args.batches:
            reference={}
            for device in devices:
                op=grtm.SurfaceWave(cohort[:b],device=device,parameterization='thickness',threads=1 if device.startswith('cuda') else args.threads)
                for kind in ('phase','group'):
                    for gradient in (False,True):
                        fn=(lambda:op.sensitivity(periods,kind=kind,parameter='velocity_s')['kernel']) if gradient else (lambda:op.dispersion(periods,kind=kind)['velocity'])
                        timing,value=measure(fn,device,args.repeat);key=(kind,gradient)
                        if key not in reference:reference[key]=array(value)
                        error=relative(value,reference[key])
                        if error>(2e-6 if device=='mps' else 1e-7):raise AssertionError(('surface mismatch',error))
                        emit(dict(family='surface',operation=kind+('_forward_vs_kernel' if gradient else '_forward'),layers=n,batch=b,backend=device,threads=op.threads,error=error,compute_device=op.device,root_tolerance_km_s=op.tolerance,**timing))
            if args.disba:
                from disba import PhaseDispersion,GroupDispersion,PhaseSensitivity,GroupSensitivity
                for kind in ('phase','group'):
                    forward_cls=PhaseDispersion if kind=='phase' else GroupDispersion
                    kernel_cls=PhaseSensitivity if kind=='phase' else GroupSensitivity
                    forwards=[forward_cls(*a.T) for a in cohort[:b]]
                    kernels=[kernel_cls(*a.T) for a in cohort[:b]]
                    for gradient in (False,True):
                        def baseline():
                            if gradient:return np.stack([np.stack([k(float(t),parameter='velocity_s').kernel for t in periods]) for k in kernels])
                            return np.stack([f(periods).velocity for f in forwards])
                        timing,value=measure(baseline,'cpu',args.repeat)
                        emit(dict(family='surface',operation=kind+('_forward_vs_kernel' if gradient else '_forward'),layers=n,batch=b,backend='disba-cpu',threads=1,version=__import__('disba').__version__,error_vs_grtm=relative(value,reference[(kind,gradient)]),algorithm='unchanged default Dunkin; dc=.005, group dt=.025, kernel dp=.025',**timing))
    if 'green' in args.families:
        n=args.green_layers;moment=np.asarray(grtm.strike_dip_rake(20,40,60,1e15));cohort=layers(n,max(args.batches));cohort[:,:,:4]*=1000
        model=grtm.example_model(units='si');model.update(layers=cohort[0],source_depth=8000.,log2_samples=6,dt=.4,distances=[10000.,20000.,30000.,50000.],t0=[0.]*4,ptam=0)
        fixed=grtm.Solver().operator(model,threads=args.threads).freeze_grid(cohort[0]);cot=np.random.default_rng(23).normal(size=(4,64,3))
        for b in [b for b in args.batches if b<=args.green_max_batch]:
            reference={}
            for device in devices:
                workers=1 if b==1 else min(4,b);threads=max(1,args.threads//workers)
                solver=grtm.Solver(backend='metal' if device=='mps' else 'cuda' if device.startswith('cuda') else 'cpu')
                op=solver.operator(model,parameters=['vp','vs'],workers=workers,threads=threads,grid=fixed,vjp_method='auto',device=torch.device(device).index or 0 if device.startswith('cuda') else 0)
                # Fixed geometry/grid, explicit host model workers; not a fused model kernel.
                for gradient in (False,True):
                    def one(i):
                        linear=op.linearize(cohort[i],moment)
                        return linear.vjp(cot,moment_gradient=False)[0] if gradient else linear.value
                    def fn():
                        if workers==1:return np.stack([one(i) for i in range(b)])
                        with ThreadPoolExecutor(workers) as pool:return np.stack(list(pool.map(one,range(b))))
                    timing,value=measure(fn,device,args.repeat)
                    if gradient not in reference:reference[gradient]=value
                    error=relative(value,reference[gradient])
                    if error>2e-7:raise AssertionError(('green mismatch',error))
                    emit(dict(family='green',operation='forward_vjp' if gradient else 'forward',layers=n,batch=b,backend=device,workers=workers,threads_per_worker=threads,error=error,model_batch='host workers',derivative='CUDA analytic directional contractions; CPU precision exceptions' if device.startswith('cuda') else 'CPU adjoint',wavenumber_nodes=int(np.sum(fixed['steps']+1)),**timing))
    report['completed']=True;report['end']=datetime.datetime.now(datetime.timezone.utc).isoformat();out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
