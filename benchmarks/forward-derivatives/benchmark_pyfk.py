"""Unmodified pyfk calculate_gf versus GRTM force Green functions.

Both use the same model, receivers, damping, taper and time origins. pyfk uses
midpoint k nodes; GRTM uses its original trapezoid rule, with the same spacing
and conservative matching cutoff. Refinement errors are reported, not fitted
away. The force channel/sign mapping follows the source equations, not data.
"""
import argparse,datetime,json,platform,statistics,time,warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyfk
import grtm


def run_time(fn,repeat):
    result=fn();ts=[]
    for _ in range(repeat):
        t=time.perf_counter();result=fn();ts.append(time.perf_counter()-t)
    return dict(seconds=statistics.median(ts),trials=ts),result

def solid(n,b):
    x=np.linspace(0,1,n);one=np.stack((45*x,6.1+1.9*x,3.5+.95*x,2.7+.6*x,np.full(n,1e9),np.full(n,1e9)),-1)
    a=np.repeat(one[None],b,0);a[:,:,2]*=np.linspace(.99,1.01,b)[:,None];return a

def make(layers,receivers,spacing,cuda=False):
    fk=np.column_stack((np.r_[np.diff(layers[:,0]),0.],layers[:,2],layers[:,1],layers[:,3],layers[:,5],layers[:,4]))
    return pyfk.Config(model=pyfk.SeisModel(fk),source=pyfk.SourceModel(sdep=8,srcType='sf'),receiver_distance=receivers,npt=64,dt=.4,dk=spacing,kmax=15,samples_before_first_arrival=5,cuda=cuda)
def py_values(config):
    streams=pyfk.calculate_gf(config)
    a=np.array([[t.data for t in s] for s in streams]).transpose(0,2,1)
    # GRTM [R_horizontal,R_vertical,T_horizontal,Z_horizontal,Z_vertical].
    return a[:,:,[4,1,5,3,0]]*np.array([1.,-1.,-1.,1.,-1.]),[s[0].stats.sac.b for s in streams]
def grtm_config(layers,receivers,spacing,shifts):
    model=dict(layers=layers,source_depth=8,receiver_depth=0,distances=receivers,t0=shifts,log2_samples=6,dt=.4,source_type=0,taper=.3,ptam=0)
    length=2*max(8,max(receivers))/spacing;dk=2*np.pi/length
    source_row=np.searchsorted(layers[:,0],8,side='right')-1
    w=2*np.pi*np.arange(32)/(64*.4);steps=np.ceil(np.sqrt((15/8)**2+(w/layers[source_row,2])**2)/dk).astype(np.int32)
    return model,dict(steps=steps,period_length=length)
def error(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--backend',default='metal',choices=['metal','cuda','cpu']);p.add_argument('--device',type=int,default=0);p.add_argument('--threads',type=int,default=16);p.add_argument('--repeat',type=int,default=3);p.add_argument('--pyfk-cuda',action='store_true');p.add_argument('--profile',choices=['throughput','refined'],default='throughput');args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    report=dict(grtm_version=grtm.__version__,pyfk_commit='810e77dee6e7fe12ebac41832dfc17e6b7bab29f',python=platform.python_version(),platform=platform.platform(),arguments=vars(args),records=[],accuracy=[],completed=False,start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save(row,kind='records'):report[kind].append(row);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    # Orthogonal small/batch, receiver and layer cases, all return five force channels.
    spacing=.0125 if args.profile=='refined' else .1
    cases=[(4,1,4),(4,16,4),(32,1,4),(32,16,4)] if args.profile=='refined' else [(4,1,4),(4,16,4),(4,64,4),(4,1,64),(32,1,4),(32,16,4)]
    for nl,b,nd in cases:
        models=solid(nl,64)[:b];receivers=np.linspace(10,50,nd);cfg=[make(a,receivers,spacing) for a in models]
        def fk_run():return [py_values(c)[0] for c in cfg]
        # Travel times are taken from the unchanged pyfk model, not fitted.
        warm=[py_values(c) for c in cfg];native=[grtm_config(a,receivers,spacing,w[1]) for a,w in zip(models,warm)]
        timing,ref=run_time(fk_run,args.repeat);ref=np.stack(ref)
        save(dict(family='green',implementation='pyfk-cpu',layers=nl,batch=b,receivers=nd,threads=1,models_per_second=b/timing['seconds'],**timing))
        for backend,workers,threads in [('cpu',1,1),('cpu',min(4,b),max(1,args.threads//min(4,b)))]+([] if args.backend=='cpu' else [(args.backend,min(4,b),max(1,args.threads//min(4,b)))]):
            solver=grtm.Solver(backend=backend)
            def one(i):return solver.compute(native[i][0],grid=native[i][1],threads=threads,device=args.device,derivatives=0)['signal'][...,:5]
            def run():
                if workers==1:return np.stack([one(i) for i in range(b)])
                with ThreadPoolExecutor(workers) as pool:return np.stack(list(pool.map(one,range(b))))
            timing,value=run_time(run,args.repeat)
            save(dict(family='green',implementation='grtm-'+backend,layers=nl,batch=b,receivers=nd,workers=workers,threads_per_worker=threads,models_per_second=b/timing['seconds'],pyfk_relative_error=error(value,ref),**timing))
        if args.pyfk_cuda:
            for c in cfg:c.cuda=True
            try:
                timing,value=run_time(fk_run,args.repeat);save(dict(family='green',implementation='pyfk-cuda',layers=nl,batch=b,receivers=nd,error=error(np.stack(value),ref),**timing))
            except Exception as exc:
                save(dict(implementation='pyfk-cuda',layers=nl,batch=b,receivers=nd,status='unavailable',error_type=type(exc).__name__,reason=str(exc)))
                args.pyfk_cuda=False
    for nl in (4,32):
        model=solid(nl,1)[0];receivers=np.array([10.,30.,50.]);previous=None
        for spacing in ((.0125,.00625,.003125) if args.profile=='refined' else (.1,.05,.025,.0125)):
            ref,shifts=py_values(make(model,receivers,spacing));m,g=grtm_config(model,receivers,spacing,shifts)
            val=grtm.Solver().compute(m,grid=g,threads=args.threads,derivatives=0)['signal'][...,:5]
            row=dict(layers=nl,spacing=spacing,pyfk_vs_grtm_relative=error(val,ref),channel_relative=[error(val[...,c],ref[...,c]) for c in range(5)])
            if previous is not None:row.update(pyfk_refinement=error(ref,previous[0]),grtm_refinement=error(val,previous[1]))
            save(row,'accuracy');previous=(ref,val)
    report['completed']=True;report['end']=datetime.datetime.now(datetime.timezone.utc).isoformat();out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
