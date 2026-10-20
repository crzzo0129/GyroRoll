"""Record original GPU policy rollouts without initializing any graphics context."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def main():
    import torch
    from .__main__ import parser,build
    from miniwheelbot_rl.task import heading_rate
    p=argparse.ArgumentParser();p.add_argument('--name',choices=('balanced','balanced_mirror'),required=True)
    p.add_argument('--stage',type=int,choices=(1,2),required=True)
    a=p.parse_args()
    root=Path('logs/steer_v2')/a.name/f'stage{a.stage}'
    checkpoint=root/('model_1999.pt' if a.stage==1 else 'model_3998.pt')
    args=parser().parse_args(['evaluate','--checkpoint',str(checkpoint),'--duration','16','--num-envs','1'])
    env,wrapped,runner,_=build(args)
    out=Path('results/raked_steer/trajectories');out.mkdir(parents=True,exist_ok=True)
    policy=runner.get_inference_policy(device=env.device)
    fps=24;states=[];metrics=[];report={'name':a.name,'stage':a.stage,'checkpoint':str(checkpoint),
        'fps':fps,'initial_speed_m_s':0.,'initial_roll_deg':0.,'seed':42,'action_smoothing':1.,
        'physics_source':'original MJLab 1.3.0 / GPU MuJoCo-Warp checkpoint rollout; saved-pose replay is rendering only',
        'columns':['case_index','time_s','target_v','actual_v','yaw_rate','roll_deg','terminated','failed_at'], 'cases':[]}
    cases=[('FORWARD',10.,lambda t:1.),('REVERSE',12.,lambda t:-1.),
           ('FORWARD TO REVERSE',15.,lambda t:1. if t<5 else -1.),('STOP',8.,lambda t:0.)]
    try:
        env.scene.write(out/'scene')
        for case,(title,duration,command) in enumerate(cases):
            env._initial_speed=0.;env._evaluation_roll=torch.zeros(1,device=env.device)
            env.reset();failed=False;failed_time=-1.;frames=0
            for step in range(math.ceil(duration/env.step_dt)):
                t=step*env.step_dt;target=command(t)
                if not failed:
                    env.command_manager.get_command('twist')[:]=torch.tensor([target,0.,0.],device=env.device)
                    with torch.inference_mode():action=policy(wrapped.get_observations()).clamp(-1,1)
                    robot=env.scene['robot'];g=robot.data.projected_gravity_b[0]
                    measured=float(robot.data.root_link_lin_vel_b[0,0]);yaw=float(heading_rate(env)[0])
                    roll=float(torch.atan2(-g[1],-g[2]).rad2deg())
                if t+1e-9>=frames/fps:
                    if not failed:last=env.sim.data.qpos[0].cpu().numpy().copy()
                    states.append(last.copy());metrics.append([case,t,target,measured,yaw,roll,float(failed),failed_time]);frames+=1
                if not failed:
                    with torch.inference_mode():_,_,done,_=wrapped.step(action)
                    if bool(done.any()):failed=True;failed_time=(step+1)*env.step_dt
            report['cases'].append({'name':title,'duration_s':duration,'frames':frames,'failed_at_s':failed_time if failed else None})
            print(a.name,a.stage,title,'frames',frames,'failed_at',failed_time,flush=True)
        stem=out/f'{a.name}_stage{a.stage}'
        np.savez_compressed(stem.with_suffix('.npz'),qpos=np.stack(states),metrics=np.array(metrics))
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:env.close()


if __name__=='__main__':main()
