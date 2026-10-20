"""Check fixed +1/-1 flywheel coupling with physical joint equality constraints.

Never overwrites simulated joint positions or velocities to fake synchronization.
MuJoCo solves compliant equality constraints: report residuals explicitly.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from probe_structures import Plant, ScheduledFeedback, DT

OUT=Path(__file__).resolve().parent/'results/twin_flywheel/coupling'


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    feedback=ScheduledFeedback(Plant('twin',coupling=1))
    results=[]
    for sign,label in ((1,'equal'),(-1,'opposite')):
        p=Plant('twin',coupling=sign)
        (OUT/(label+'.xml')).write_text(p.xml,encoding='utf-8')
        for dt in (.002,.001,.0005):
            p.m.opt.timestep=dt
            for initial_roll in (-1.,1.):
                for speed in (0.,1.):
                    feedback.reset()
                    d=p.reset(roll=np.radians(initial_roll))
                    rows=[];peak_residual=0.;elapsed=0.;failed=False;target_speed=0.
                    for step in range(1500):
                        x=p.state(d)
                        if abs(x[0])>.7 or abs(x[1])>.7 or d.qpos[2]<.22:
                            failed=True;break
                        target_speed+=np.clip(speed-target_speed,-.5*DT,.5*DT)
                        # For opposite mode this is deliberately the same attempted
                        # roll correction: the gearing cancels the direct roll channel.
                        p.input(d,feedback.act(x,target_speed,0.))
                        for _ in range(round(DT/dt)):
                            mujoco.mj_step(p.m,d)
                            error=d.qvel[p.dofs[1]]-sign*d.qvel[p.dofs[2]]
                            peak_residual=max(peak_residual,abs(float(error)))
                        mujoco.mj_forward(p.m,d)
                        elapsed=(step+1)*DT
                        rows.append([elapsed,*d.qvel[p.dofs[1:]],float(error),p.state(d)[0]])
                    result=dict(mode=label,relation=f'omega_left = {sign} * omega_right',
                        dt=dt,initial_roll_deg=initial_roll,command_speed=speed,
                        survived=not failed,duration_s=elapsed,max_constraint_speed_error_rad_s=peak_residual,
                        peak_flywheel_speed_rad_s=float(np.max(abs(np.array(rows)[:,1:3]))))
                    results.append(result)
                    if dt==.002 and initial_roll==1.:
                        np.savetxt(OUT/f'{label}_{speed:g}.csv',rows,delimiter=',',
                            header='time_s,omega_left,omega_right,relation_residual,roll_rad',comments='')
                    print(json.dumps(result),flush=True)
    report={'axes':'Both axes use chassis +x; speeds are joint speeds relative to the chassis.',
        'constraint':'q_left = sign*q_right; zero initial joint angles; fixed mode per rollout',
        'limitation':'MuJoCo joint equalities are compliant numerical constraints, not bit-exact velocity identities.',
        'control':'Two independent generalized commands: drive and coupled rotor; motor torque split [tau/2, sign*tau/2]. Opposite mode cancels direct roll authority; no claim of an optimized opposite-mode controller.',
        'cases':results}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':run()
