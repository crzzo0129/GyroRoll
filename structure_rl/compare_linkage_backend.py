"""Compare the same initial state and controls in native MuJoCo and MJLab/Warp."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import mujoco
    import numpy as np
    import torch
    from .__main__ import parser,build
    from .linkage_seed import affine_seed
    args=parser().parse_args(['evaluate','--checkpoint',str(a.checkpoint),'--num-envs','1'])
    env,wrapped,runner,_=build(args)
    try:
        env._evaluation_roll=torch.tensor([np.deg2rad(.1)],device=env.device)
        env._evaluation_pitch=torch.tensor([np.deg2rad(-.1)],device=env.device)
        env.reset()
        model=env.sim.mj_model
        cpu=mujoco.MjData(model)
        cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
        cpu.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model,cpu)
        policy=runner.get_inference_policy(device=env.device)
        weight,bias=affine_seed()
        rows=[]
        for step in range(30):
            env.command_manager.get_command('twist')[:]=0.
            with torch.inference_mode():
                obs=wrapped.get_observations()
                action=policy(obs).clamp(-1,1)
                expected=torch.tanh(obs['actor'].cpu()@weight.T+bias)
                _,_,done,_=wrapped.step(action)
            ctrl=env.sim.data.ctrl[0].cpu().numpy()
            cpu.ctrl[:]=ctrl
            mujoco.mj_step(model,cpu,nstep=round(env.step_dt/model.opt.timestep))
            gpu=env.sim.data.qpos[0].cpu().numpy()
            rows.append({'t':(step+1)*env.step_dt,'obs':obs['actor'][0].cpu().tolist(),
                'action':action[0].cpu().tolist(),'expected':expected[0].tolist(),'ctrl':ctrl.tolist(),
                'cpu_qpos':cpu.qpos.tolist(),'gpu_qpos':gpu.tolist(),'max_qpos_difference':float(np.max(abs(cpu.qpos-gpu)))})
            if bool(done.any()):break
        report={'rows':rows,'timestep':model.opt.timestep,'cone':int(model.opt.cone),
            'actuator_gainprm':model.actuator_gainprm.tolist(),'actuator_biasprm':model.actuator_biasprm.tolist(),
            'qpos0':model.qpos0.tolist()}
        a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
        for row in rows[::5]:print(json.dumps(row),flush=True)
    finally:env.close()


if __name__=='__main__':
    main()
