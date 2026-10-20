"""Inspect deterministic linkage policy failures without a graphics context."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from .__main__ import parser,build
    args=parser().parse_args(['evaluate','--checkpoint',str(a.checkpoint),'--num-envs','1','--duration','10'])
    env,wrapped,runner,_=build(args)
    policy=runner.get_inference_policy(device=env.device)
    robot=env.scene['robot']
    ids=robot.find_joints(('drive','linkage_motor','wheel_roll','passive_rod'),preserve_order=True)[0]
    results=[]
    try:
        for control in ('policy','zero'):
            for roll_deg in (0.,.1,-.1):
                env._initial_speed=0.
                env._evaluation_roll=torch.tensor([math.radians(roll_deg)],device=env.device)
                env.reset()
                rows=[]
                for step in range(round(10/env.step_dt)):
                    env.command_manager.get_command('twist')[:]=0.
                    with torch.inference_mode():
                        action=policy(wrapped.get_observations()).clamp(-1,1) if control=='policy' else torch.zeros((1,2),device=env.device)
                        _,_,done,_=wrapped.step(action)
                    if step%10==0 or bool(done.any()):
                        g=robot.data.projected_gravity_b[0]
                        rows.append({'time':(step+1)*env.step_dt,'height':float(robot.data.root_link_pos_w[0,2]),
                            'roll_deg':float(torch.atan2(-g[1],-g[2]).rad2deg()),
                            'pitch_deg':float(torch.atan2(g[0],torch.linalg.vector_norm(g[1:])).rad2deg()),
                            'action':action[0].tolist(),'joint_pos':robot.data.joint_pos[0,ids].tolist(),
                            'joint_vel':robot.data.joint_vel[0,ids].tolist(),
                            'force':robot.data.actuator_force[0].tolist(),
                            'linear_velocity':robot.data.root_link_lin_vel_b[0].tolist(),'terminated':bool(done.any())})
                    if bool(done.any()):break
                item={'control':control,'initial_roll_deg':roll_deg,'duration':(step+1)*env.step_dt,'rows':rows}
                results.append(item)
                print(json.dumps({'control':control,'roll':roll_deg,'first':rows[0],'last':rows[-1]}),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'checkpoint':str(a.checkpoint),'cases':results},indent=2),encoding='utf-8')
    finally:
        env.close()


if __name__=='__main__':
    main()
