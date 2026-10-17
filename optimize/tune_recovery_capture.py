"""Tune bounded nonlinear PD capture after a saved recovery trajectory."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from probe_rise import Experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/chamfer_recovery/tuned.json'))
    p.add_argument('--trials',type=int,default=320)
    p.add_argument('--seed',type=int,default=93)
    args=p.parse_args()
    config=json.loads(args.controller.read_text())
    ex=Experiment()
    m=ex.model
    base=ex.reset(settle=config.get('settle_s',2.))
    for duration,f,d in config['schedule']:
        base.ctrl[:]=[f,d]
        mujoco.mj_step(m,base,nstep=round(duration/ex.settings.timestep))
    limits=np.array([250.,35.])
    rng=np.random.default_rng(args.seed)
    bounds=np.array([[6,45],[.8,8],[0,.03],[6,30],[.7,5],[0,3],[-1,2],[-1,2]])
    best=1e9
    best_x=np.array([16,3,.004,10,1.5,1.5,0,0])
    history=[]
    for i in range(args.trials):
        if i==0:
            x=best_x.copy()
        elif i%4:
            x=np.clip(best_x+rng.normal(size=8)*(bounds[:,1]-bounds[:,0])*rng.choice([.03,.1,.25]),bounds[:,0],bounds[:,1])
        else:
            x=rng.uniform(bounds[:,0],bounds[:,1])
        d=mujoco.MjData(m)
        mujoco.mj_copyData(d,m,base)
        kr,dr,hr,kp,dp,vp,gr,gp=x
        cost,stable,maxstable=0.,0.,0.
        for k in range(400):
            phi,theta,w,v,tilt=ex.state(d)
            speeds=d.qvel[ex.dofs]
            tq=np.array([kr*phi+dr*w[0]+hr*speeds[0]+gr*.003574476*speeds[1]*w[2],
                         kp*theta+dp*w[1]+vp*v[0]-gp*.00151994*speeds[0]*w[2]])
            d.ctrl[:]=np.clip(speeds+tq/[.12,.25],-limits,limits)
            stable=stable+.01 if tilt<np.radians(20) and d.qpos[2]>.3 and np.linalg.norm(w[:2])<1.5 else 0.
            maxstable=max(stable,maxstable)
            if k>=250:
                cost+=tilt**2+.02*np.dot(w[:2],w[:2])+.000001*speeds[0]**2
            before=d.time
            mujoco.mj_step(m,d,nstep=5)
            if d.time<=before or not np.isfinite(d.qpos).all():
                cost=1e9
                break
        score=cost*.01-3*maxstable-5*stable
        history.append(dict(trial=i,score=float(score),max_stable_s=maxstable,final_stable_s=stable))
        if score<best:
            best,best_x=score,x.copy()
            result=dict(config,params=x[:6].tolist(),roll_gyro_gain=gr,gyro_gain=gp,
                        tuning_score=float(best),tuning_max_stable_s=maxstable,tuning_final_stable_s=stable)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(history[-1]),flush=True)
        if i%40==0: print(f'Tuned {i}/{args.trials}',flush=True)
    args.output.with_suffix('.search.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


if __name__=='__main__':main()
