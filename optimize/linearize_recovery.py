"""Build time-varying LQR feedback around a bounded recovery trajectory."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from probe_rise import Experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/wheelbot_cargo/chamfer_recovery/reference.npz'))
    args=p.parse_args()
    c=json.loads(args.controller.read_text())
    ex=Experiment()
    m=ex.model
    d=ex.reset(settle=c.get('settle_s',2.))
    targets=np.concatenate([np.tile([f,w],(round(t/ex.dt),1)) for t,f,w in c['schedule']])
    refs,vels,As,Bs=[],[],[],[]
    for i,target in enumerate(targets):
        refs.append(d.qpos.copy())
        vels.append(d.qvel.copy())
        d.ctrl[:]=target
        A=np.eye(2*m.nv)
        B=np.zeros((2*m.nv,m.nu))
        for _ in range(ex.settings.decimation):
            a,b=np.empty_like(A),np.empty_like(B)
            mujoco.mjd_transitionFD(m,d,1e-5,True,a,b,None,None)
            A,B=a@A,a@B+b
            mujoco.mj_step(m,d)
        As.append(A)
        Bs.append(B)
        if i%80==0: print(f'Linearized {i}/{len(targets)}',flush=True)
    # Free-joint tangent coordinates: xyz, local rotation, drive/reaction angles.
    # Wheel angles do not define balance; prioritize attitude and body rates.
    qdiag=np.array([.2,.2,20,35,35,1,0,0, .2,.2,.2,2,2,.1,.0001,.0001])
    Q=np.diag(qdiag)*ex.dt
    R=np.diag([1/250**2,1/35**2])*ex.dt
    P=np.diag(qdiag)*3
    gains=[]
    for A,B in zip(As[::-1],Bs[::-1]):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        P=Q+A.T@P@(A-B@K)
        P=(P+P.T)*.5
        gains.append(K)
    np.savez_compressed(args.output,qpos=refs,qvel=vels,targets=targets,
                        gains=gains[::-1],control_dt=ex.dt,
                        geometry_sha256=c.get('geometry_sha256',''))
    print(args.output,flush=True)


if __name__=='__main__':main()
