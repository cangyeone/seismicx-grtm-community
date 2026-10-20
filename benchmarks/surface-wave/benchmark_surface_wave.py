"""Synchronized end-to-end GRTM CUDA kernels vs warmed disba CPU kernels.

disba is a benchmark/reference dependency only, never a GRTM backend.
"""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import torch
from disba import PhaseSensitivity,GroupSensitivity
from grtm import SurfaceWave

def model(n):
    h=np.r_[np.linspace(.5,1.8,n-1),0.];vs=np.linspace(1.9,4.4,n);vp=1.8*vs;rho=np.linspace(2.2,3.2,n)
    vs[-1]=4.6;vp[-1]=8.1;rho[-1]=3.3
    return np.stack((h,vp,vs,rho),-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--layers',nargs='+',type=int,default=[8,32]);p.add_argument('--batches',nargs='+',type=int,default=[1,16,64]);p.add_argument('--periods',type=int,default=12);p.add_argument('--repeat',type=int,default=2);p.add_argument('--kinds',nargs='+',default=['phase','group']);p.add_argument('--wave',default='rayleigh');p.add_argument('--parameter',default='velocity_s');p.add_argument('--baseline',choices=['disba','grtm'],default='disba');p.add_argument('--output',default='surface-wave-benchmark.json');a=p.parse_args()
    torch.set_num_threads(1);periods=np.geomspace(2,50,a.periods);records=[]
    metadata={'gpu':torch.cuda.get_device_name(a.device),'device':a.device,'cpu':platform.processor(),'host':platform.node(),'torch':torch.__version__,'numpy':np.__version__,'cpu_baseline':'disba 0.7.0, Dunkin, dc=.005, dp=.025, group dt=.025','protocol':'Prepared operators; host wall time including native root search, derivatives and device synchronization. Model transfer/operator preparation reported separately. Numba and CUDA warmed. Repetitions are whole batches. No CPU loop or disba in GPU target.','periods':periods.tolist(),'records':records}
    metadata['shared_hardware']=True
    metadata['protocol']='Prepared operators, both warmed; synchronized host wall time includes native root search and derivatives. Model transfer/operator setup reported separately. Numba compilation excluded when disba is used. Each repetition is a complete model batch.'
    if a.baseline=='grtm':metadata['cpu_baseline']='GRTM native CPU, identical semi-analytic mathematics and adaptive search grid, one C++ thread'
    for n in a.layers:
        for batch in a.batches:
            m=np.repeat(model(n)[None],batch,0);m[:,:,2]*=np.linspace(.99,1.01,batch)[:,None]
            for kind in a.kinds:
                start=time.perf_counter();cls=PhaseSensitivity if kind=='phase' else GroupSensitivity;refs=[cls(*mi.T,algorithm='dunkin') for mi in m] if a.baseline=='disba' else SurfaceWave(m,parameterization='thickness');cpu_setup=time.perf_counter()-start
                start=time.perf_counter();op=SurfaceWave(torch.tensor(m,dtype=torch.float64,device=a.device),parameterization='thickness');torch.cuda.synchronize(a.device);gpu_setup=time.perf_counter()-start
                def baseline():
                    if a.baseline=='grtm':return refs.sensitivity(periods,wave=a.wave,kind=kind,parameter=a.parameter)['kernel']
                    return np.stack([[ref(float(t),wave=a.wave,parameter=a.parameter).kernel for t in periods] for ref in refs])
                def accelerated():return op.sensitivity(periods,wave=a.wave,kind=kind,parameter=a.parameter)
                # Same kernel/model shape is warmed before recorded repetitions.
                if a.baseline=='disba':refs[0](float(periods[0]),wave=a.wave,parameter=a.parameter)
                else:baseline()
                accelerated();torch.cuda.synchronize(a.device)
                cpu=[];gpu=[]
                for _ in range(a.repeat):
                    start=time.perf_counter();expected=baseline();cpu.append(time.perf_counter()-start)
                    torch.cuda.synchronize(a.device);start=time.perf_counter();actual=accelerated();torch.cuda.synchronize(a.device);gpu.append(time.perf_counter()-start)
                difference=actual['kernel'].detach().cpu().numpy()-expected
                r=dict(layers=n,batch=batch,kind=kind,wave=a.wave,parameter=a.parameter,period_count=len(periods),cpu_seconds=cpu,cuda_seconds=gpu,cpu_median=float(np.median(cpu)),cuda_median=float(np.median(gpu)),speedup=float(np.median(cpu)/np.median(gpu)),cpu_setup=cpu_setup,cuda_setup=gpu_setup,max_disba_kernel_difference=float(np.abs(difference).max()),kernel_difference_note='disba uses a one-sided 2.5% model perturbation; agreement with this approximation is not the derivative-accuracy test')
                records.append(r)
                r['baseline']=a.baseline
                if a.baseline=='grtm':
                    r['max_native_cpu_kernel_difference']=r.pop('max_disba_kernel_difference');r['kernel_difference_note']='Same mathematical implementation; CPU/CUDA numerical parity'
                Path(a.output).write_text(json.dumps(metadata,indent=2));print(json.dumps(r),flush=True)
    print('Saved',a.output,flush=True)
if __name__=='__main__':main()
