"""Experimental state feedback for a locally planned get-up trajectory.

This is separate from the RL policy. References must match the robot geometry.
"""
from pathlib import Path
import hashlib
import mujoco
import numpy as np


class RecoveryTrajectory:
    def __init__(self, path, model, model_path, scale=1., side=1.):
        with np.load(path) as data:
            self.qpos=data['qpos'].copy()
            self.qvel=data['qvel'].copy()
            self.targets=data['targets'].copy()
            self.gains=data['gains'].copy()
            self.dt=float(data['control_dt'])
            digest=str(data['geometry_sha256'])
        if digest and digest != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
            raise ValueError('Recovery reference does not match robot geometry')
        self.model=model
        self.scale=scale
        self.side=1. if side>=0 else -1.
        self.rotor_dof=int(model.joint('reaction').dofadr[0])
        self.rotor_qpos=int(model.joint('reaction').qposadr[0])

    def control(self, data, elapsed):
        i=round(elapsed/self.dt)
        if i>=len(self.targets):
            return None
        qpos,qvel=data.qpos.copy(),data.qvel.copy()
        if self.side<0:
            qpos[1]*=-1
            qpos[[4,6]]*=-1
            qpos[self.rotor_qpos]*=-1
            qvel[[1,3,5,self.rotor_dof]]*=-1
        delta=np.empty(self.model.nv)
        mujoco.mj_differentiatePos(self.model,delta,1.,self.qpos[i],qpos)
        error=np.r_[delta,qvel-self.qvel[i]]
        target=self.targets[i]-self.scale*(self.gains[i]@error)
        target[0]*=self.side
        return np.clip(target, self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])


class PredictiveRecovery(RecoveryTrajectory):
    """Replan short wheel-speed segments from the measured simulator state.

    Intended for local feasibility experiments; no real-time latency guarantee.
    Uses the unchanged velocity servos and contact model for every prediction.
    """
    def __init__(self,*args, population=48, iterations=4, threads=6, seed=71, **kwargs):
        super().__init__(*args,**kwargs)
        from mujoco import rollout
        self.pool=rollout.Rollout(nthread=threads)
        self.datas=[mujoco.MjData(self.model) for _ in range(threads)]
        self.population=population
        self.iterations=iterations
        self.rng=np.random.default_rng(seed)
        self.hold=.04
        self.horizon=10
        self.last_tick=-1
        self.last_target=None
        self.previous=None
        self.records=[]
        if self.side<0:
            self.qpos[:,1]*=-1
            self.qpos[:,[4,6]]*=-1
            self.qpos[:,self.rotor_qpos]*=-1
            self.qvel[:,[1,3,5,self.rotor_dof]]*=-1
            self.targets[:,0]*=-1

    def close(self):
        self.pool.close()

    def control(self,data,elapsed):
        if round(elapsed/self.dt)>=len(self.targets):
            return None
        tick=int(round(elapsed/self.dt))//round(self.hold/self.dt)
        if tick==self.last_tick:
            return self.last_target.copy()
        self.last_tick=tick
        m=self.model
        repeat=round(self.hold/m.opt.timestep)
        future=elapsed+np.arange(1,self.horizon*repeat+1)*m.opt.timestep
        indices=np.minimum(np.round(future/self.dt).astype(int),len(self.qpos)-1)
        qref,vref=self.qpos[indices],self.qvel[indices]
        knots=np.minimum(np.round((elapsed+np.arange(self.horizon)*self.hold)/self.dt).astype(int),len(self.targets)-1)
        nominal=self.targets[knots]/[250.,35.]
        mean=nominal.copy() if self.previous is None else np.concatenate([self.previous[1:],nominal[-1:]])
        std=np.full(mean.shape,.18)
        spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
        initial=np.empty(mujoco.mj_stateSize(m,spec))
        mujoco.mj_getState(m,data,initial,spec)
        best,best_x=float('inf'),mean.copy()
        for iteration in range(self.iterations):
            x=np.clip(self.rng.normal(mean,std,(self.population,self.horizon,2)),-1,1)
            x[0],x[1],x[2]=best_x,nominal,mean
            controls=np.repeat(x*[250.,35.],repeat,axis=1)
            states,_=self.pool.rollout(m,self.datas,initial,controls,initial_warmstart=data.qacc_warmstart.copy())
            qs=states[...,1:1+m.nq]
            vs=states[...,1+m.nq:]
            dot=np.abs(np.sum(qs[...,3:7]*qref[None,:,3:7],axis=-1))
            orientation=1-np.clip(dot,0,1)**2
            cost=20*orientation+4*np.sum((qs[...,:3]-qref[None,:,:3])**2,axis=-1)
            cost+=.06*np.sum((vs[...,3:6]-vref[None,:,3:6])**2,axis=-1)
            cost+=.05*np.sum((vs[...,:3]-vref[None,:,:3])**2,axis=-1)
            cost+=.000005*(vs[...,self.rotor_dof]-vref[None,:,self.rotor_dof])**2
            scores=cost.mean(axis=1)+cost[:,-repeat:].mean(axis=1)
            scores[np.any(np.diff(states[...,0],axis=1)<=0,axis=1)]=1e9
            order=np.argsort(scores)
            if scores[order[0]]<best:
                best=float(scores[order[0]])
                best_x=x[order[0]].copy()
            elites=x[order[:max(4,self.population//8)]]
            mean=.2*mean+.8*elites.mean(axis=0)
            std=np.maximum(.015,.2*std+.8*elites.std(axis=0))
        self.previous=best_x
        self.last_target=best_x[0]*[250.,35.]
        self.records.append(dict(time=elapsed,cost=best,target=self.last_target.tolist()))
        return self.last_target.copy()
