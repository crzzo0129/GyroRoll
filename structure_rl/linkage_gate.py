"""Independent deterministic trials for curriculum promotion; failures cannot pass."""
import argparse
import json
import math
from pathlib import Path

from .linkage_training import gate_cases,promotion_passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--phase',choices=('warmup','moving','tracking'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--motion-fraction',type=float,default=1.)
    p.add_argument('--airborne-reset',action='store_true',help='Also test release from 1 mm above nominal ground contact')
    p.add_argument('--affine-reference',action='store_true',help='Diagnostic reference only; never eligible for curriculum promotion')
    a=p.parse_args()
    if not math.isfinite(a.motion_fraction) or not 0<a.motion_fraction<=1:
        raise ValueError('motion-fraction must be in (0,1]')
    if a.phase!='moving' and a.motion_fraction!=1:
        raise ValueError('Partial lessons apply only to moving')
    import torch
    from .__main__ import parser,build
    from .task import metric
    from miniwheelbot_rl.task import heading_rate
    cases=gate_cases(a.phase);trials=16;count=len(cases)*trials
    if a.phase=='moving':
        cases=[(name,speed*a.motion_fraction,yaw) for name,speed,yaw in cases]
    duration=12. if a.phase=='warmup' else 15.
    args=parser().parse_args(['evaluate','--checkpoint',str(a.checkpoint),
        '--phase',a.phase,'--motion-fraction',str(a.motion_fraction),
        '--num-envs',str(count),'--duration',str(duration),'--seed',str(a.seed)])
    env,wrapped,runner,_=build(args)
    try:
        device=env.device;robot=env.scene['robot']
        policy=runner.get_inference_policy(device=device)
        if a.affine_reference:
            from .linkage_seed import affine_seed
            weight,bias=affine_seed(torch.float64)
            weight,bias=weight.to(device),bias.to(device)
            policy=lambda obs:torch.tanh(obs['actor'].double()@weight.T+bias).float()
        tilt=math.radians(.1 if a.phase=='warmup' else (.3 if a.phase=='moving' else 1.))
        if a.phase=='moving':
            tilt*=1/3+2/3*a.motion_fraction
        env._evaluation_roll=torch.linspace(-tilt,tilt,trials,device=device).repeat(len(cases))
        env._evaluation_pitch=torch.tensor([-tilt,tilt],device=device).repeat(count//2)
        env._initial_speed=0.
        env._airborne_reset=a.airborne_reset
        env.reset()
        command=torch.tensor([[speed,0.,yaw] for _,speed,yaw in cases],device=device).repeat_interleave(trials,dim=0)
        alive=torch.ones(count,dtype=torch.bool,device=device)
        survival=torch.full((count,),duration,device=device)
        sums=torch.zeros((count,5),device=device);samples=torch.zeros(count,device=device)
        max_roll=torch.zeros(count,device=device);max_pitch=torch.zeros_like(max_roll)
        max_closure=torch.zeros_like(max_roll)
        sat=torch.zeros_like(max_roll);action_samples=torch.zeros_like(max_roll)
        for step in range(round(duration/env.step_dt)):
            t=step*env.step_dt
            if a.phase=='tracking':
                command[-trials:,0]=.6 if t<duration/2 else -.6
            env.command_manager.get_command('twist')[:]=command
            with torch.inference_mode():
                obs=wrapped.get_observations()
                action=policy(obs).clamp(-1,1)
                finite=torch.isfinite(action).all(dim=1)&torch.isfinite(obs['actor']).all(dim=1)
                invalid=alive&~finite
                reset_invalid=invalid.clone()
                survival[invalid]=t;alive&=finite
                _,_,done,_=wrapped.step(torch.nan_to_num(action))
                roll=metric(env,'roll_abs','roll_linkage')
                pitch=metric(env,'pitch_abs','roll_linkage')
                closure=metric(env,'closure_error','roll_linkage')
                finite_state=torch.isfinite(roll)&torch.isfinite(pitch)&torch.isfinite(closure)
                invalid=alive&~finite_state
                reset_invalid|=invalid
                survival[invalid]=(step+1)*env.step_dt;alive&=finite_state
                errors=torch.stack((robot.data.root_link_lin_vel_b[:,0]-command[:,0],
                    heading_rate(env)-command[:,2],robot.data.root_link_lin_vel_b[:,1],roll,pitch),dim=1)
                invalid=alive&~torch.isfinite(errors).all(dim=1)
                reset_invalid|=invalid
                survival[invalid]=(step+1)*env.step_dt;alive&=~invalid
                if t>=1.:
                    sums+=torch.where(alive[:,None],errors.square(),0.)
                    samples+=alive
                max_roll=torch.maximum(max_roll,torch.where(alive,roll,0.))
                max_pitch=torch.maximum(max_pitch,torch.where(alive,pitch,0.))
                max_closure=torch.maximum(max_closure,torch.where(alive,closure,0.))
                sat+=torch.where(alive,(action[:,1].abs()>.98).float(),0.)
                action_samples+=alive
                failed=alive&done.bool()
                survival[failed]=(step+1)*env.step_dt
                alive&=~done.bool()
                if not bool(alive.any()):break
                reset_ids=(done.bool()|reset_invalid).nonzero(as_tuple=False).flatten()
                if len(reset_ids):env.reset(env_ids=reset_ids)
        rms=(sums/samples.clamp_min(1)[:,None]).sqrt()
        speed_limit=.12 if a.phase=='warmup' else (.2 if a.phase=='moving' else .3)
        if a.phase=='moving':
            speed_limit*=a.motion_fraction
        passed=(alive&(samples>0)&torch.isfinite(rms).all(dim=1)
            &(rms[:,0]<speed_limit)&(rms[:,1]<.15)&(rms[:,2]<.2)
            &(rms[:,3]<math.radians(5))&(rms[:,4]<math.radians(5))
            &(max_roll<math.radians(20))&(max_pitch<math.radians(20))&(max_closure<.001))
        report={'checkpoint':str(a.checkpoint),'phase':a.phase,'seed':a.seed,
            'duration_s':duration,'trials_per_case':trials,'initial_speed_m_s':0.,
            'airborne_reset':a.airborne_reset,'training_profile':args.profile,
            'motion_fraction':a.motion_fraction,
            'controller':'affine_reference' if a.affine_reference else 'neural_policy',
            'initial_tilt_bound_deg':math.degrees(tilt),
            'thresholds':{'survival_rate':.9,'pass_rate':.8,'speed_rmse':speed_limit,
                'yaw_rmse':.15,'lateral_rmse':.2,'roll_pitch_rms_deg':5.,'peak_tilt_deg':20.,
                'closure_error_m':.001},'cases':[]}
        for i,(name,speed,yaw) in enumerate(cases):
            span=slice(i*trials,(i+1)*trials)
            case={'name':name,'command':[speed,yaw],'survival_rate':float(alive[span].float().mean()),
                'pass_rate':float(passed[span].float().mean()),'mean_survival_s':float(survival[span].mean()),
                'max_closure_error_m':float(max_closure[span].max()),'individual':[]}
            for j in range(i*trials,(i+1)*trials):
                case['individual'].append({'passed':bool(passed[j]),'survived':bool(alive[j]),
                    'survival_s':float(survival[j]),'rmse':rms[j].tolist() if samples[j] else None,
                    'max_roll_deg':math.degrees(float(max_roll[j])),
                    'max_pitch_deg':math.degrees(float(max_pitch[j])),
                    'linkage_action_saturation_fraction':float(sat[j]/action_samples[j].clamp_min(1))})
            report['cases'].append(case)
        report['promote']=promotion_passed(report)
        report['lesson_passed']=promotion_passed(report,motion_fraction=a.motion_fraction)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('GATE',json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
        for case in report['cases']:print(json.dumps({k:v for k,v in case.items() if k!='individual'}),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
