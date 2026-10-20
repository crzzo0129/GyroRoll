"""Portable structure_rl video entry point: GPU rollout, then separate EGL replay.

Upload this file to the cloud repository root and run it with that environment's
Python. It uses the existing structure_rl package; no new training code is needed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Support placement either in repository root or tools/cloud.
for candidate in (Path(__file__).resolve().parent, Path.cwd(), Path(__file__).resolve().parents[2]):
    if (candidate / 'structure_rl' / '__main__.py').is_file():
        sys.path.insert(0, str(candidate))
        os.chdir(candidate)
        break


def rollout_cases(selection):
    turns = [
        ('LEFT TURN', 12., lambda t: (.6, .3)),
        ('RIGHT TURN', 12., lambda t: (.6, -.3)),
    ]
    if selection == 'turns':
        return turns
    return [
        ('FORWARD', 10., lambda t: (1., 0.)),
        ('REVERSE', 12., lambda t: (-1., 0.)),
        *turns,
        ('FORWARD TO REVERSE', 15., lambda t: (1. if t < 5 else -1., 0.)),
        ('STOP', 8., lambda t: (0., 0.)),
    ]


def record(a):
    import math
    import numpy as np
    import torch
    from structure_rl.__main__ import parser,build
    from miniwheelbot_rl.task import heading_rate
    checkpoint = a.checkpoint
    args=parser().parse_args(['evaluate','--checkpoint',str(checkpoint),'--duration','16','--num-envs','1'])
    env,wrapped,runner,_=build(args)
    out=a.output/'trajectory';out.mkdir(parents=True,exist_ok=True)
    policy=runner.get_inference_policy(device=env.device)
    fps=24;states=[];metrics=[];report={'label':a.label,'checkpoint':str(checkpoint),
        'fps':fps,'initial_speed_m_s':0.,'initial_roll_deg':0.,'seed':42,'action_smoothing':1.,
        'physics_source':'original MJLab 1.3.0 / GPU MuJoCo-Warp checkpoint rollout; saved-pose replay is rendering only',
        'columns':['case_index','time_s','target_v','actual_v','yaw_rate','roll_deg','terminated','failed_at','target_yaw_rate'], 'cases':[]}
    cases=rollout_cases(a.cases)
    try:
        env.scene.write(out/'scene')
        for case,(title,duration,command) in enumerate(cases):
            env._initial_speed=0.;env._evaluation_roll=torch.zeros(1,device=env.device)
            env.reset();failed=False;failed_time=-1.;frames=0
            for step in range(math.ceil(duration/env.step_dt)):
                t=step*env.step_dt;target,target_yaw=command(t)
                if not failed:
                    env.command_manager.get_command('twist')[:]=torch.tensor([target,0.,target_yaw],device=env.device)
                    with torch.inference_mode():action=policy(wrapped.get_observations()).clamp(-1,1)
                    robot=env.scene['robot'];g=robot.data.projected_gravity_b[0]
                    measured=float(robot.data.root_link_lin_vel_b[0,0]);yaw=float(heading_rate(env)[0])
                    roll=float(torch.atan2(-g[1],-g[2]).rad2deg())
                if t+1e-9>=frames/fps:
                    if not failed:last=env.sim.data.qpos[0].cpu().numpy().copy()
                    states.append(last.copy());metrics.append([case,t,target,measured,yaw,roll,float(failed),failed_time,target_yaw]);frames+=1
                if not failed:
                    with torch.inference_mode():_,_,done,_=wrapped.step(action)
                    if bool(done.any()):failed=True;failed_time=(step+1)*env.step_dt
            report['cases'].append({'name':title,'duration_s':duration,'frames':frames,'failed_at_s':failed_time if failed else None})
            print(title,'frames',frames,'failed_at',failed_time,flush=True)
        stem=out/'rollout'
        np.savez_compressed(stem.with_suffix('.npz'),qpos=np.stack(states),metrics=np.array(metrics))
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:env.close()



def render(a):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    source = a.output / 'trajectory'
    stem='rollout';trajectory=np.load(source/(stem+'.npz'))
    report=json.loads((source/(stem+'.json')).read_text())
    model=mujoco.MjModel.from_xml_path(str(source/'scene/scene.xml'))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=360
    # Viewer extent and camera affect appearance only; original simulation poses are reused.
    model.stat.extent=2.
    data=mujoco.MjData(model);camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=1.35;camera.azimuth=135.;camera.elevation=-20.
    base_id=model.body('robot/base').id
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    a.output.mkdir(parents=True,exist_ok=True)
    with mujoco.Renderer(model,height=360,width=640) as renderer:
        with imageio.get_writer(str(a.output/(stem+'.mp4')),fps=report['fps'],codec='libx264',quality=None,
                               ffmpeg_params=['-crf','28','-preset','fast'],macro_block_size=1) as writer:
            for index,(qpos,meta) in enumerate(zip(trajectory['qpos'],trajectory['metrics'])):
                case,t,target,measured,yaw,roll,failed,failed_time=meta[:8]
                target_yaw=float(meta[8]) if len(meta)>8 else 0.
                data.qpos[:]=qpos
                mujoco.mj_forward(model,data)  # Forward kinematics only, never mj_step.
                camera.lookat[:]=data.xpos[base_id]+np.array([0.,0.,-.06])
                renderer.update_scene(data,camera=camera)
                frame=Image.new('RGB',(640,448),(15,23,36));frame.paste(Image.fromarray(renderer.render()),(0,0))
                draw=ImageDraw.Draw(frame)
                label=report['label']
                draw.rectangle((0,0,640,30),fill=(15,23,36));draw.text((12,5),label,font=font,fill='white')
                title=report['cases'][int(case)]['name']
                draw.text((12,365),f'{title} | {t:04.1f}s | v*={target:+.1f} m/s | yaw*={target_yaw:+.2f} rad/s',font=font,fill=(120,220,255))
                draw.text((12,390),f'actual v={measured:+.2f} m/s | yaw rate={yaw:+.2f} rad/s | roll={roll:+.1f} deg',font=font,fill='white')
                state=(f'TERMINATED at {failed_time:.2f}s - last pose held' if failed else
                       ('RESET: zero speed, zero roll; no action filtering' if t<2 else 'Continuous rollout; * = commanded value'))
                draw.text((12,417),state,font=font,fill=(255,110,100) if failed else (190,200,215))
                writer.append_data(np.array(frame))
                if index in (0,300,650,950):frame.save(a.output/(stem+f'_frame{index}.png'))
                if index%240==0:print(stem,'frame',index,flush=True)
    report['rendering']=f'MuJoCo {mujoco.__version__}; EGL physical GPU {a.gpu}; recorded qpos replay, no re-simulation'
    (a.output/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VIDEO_COMPLETE',stem,flush=True)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/raked_steer/rl/mixed_maneuvers_pitch_video'))
    p.add_argument('--gpu', type=int, default=1, help='Physical GPU index for both rollout and EGL')
    p.add_argument('--label', default='Ordinary optimization')
    p.add_argument('--cases', choices=('all', 'turns'), default='all',
                   help='all: forward/reverse/left/right/reversal/stop; turns: left/right only')
    p.add_argument('--worker', choices=('record', 'render'), help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.checkpoint.is_file():
        p.error(f'Checkpoint does not exist: {a.checkpoint}')
    if not (a.checkpoint.parent / 'task.json').is_file():
        p.error('Keep task.json and geometry.xml beside the original checkpoint.')
    if a.worker == 'record':
        record(a)
    elif a.worker == 'render':
        render(a)
    else:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), MUJOCO_GL='egl',
                   PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(a.gpu))
        child = [sys.executable, '-u', str(Path(__file__).resolve()),
                 '--checkpoint', str(a.checkpoint.resolve()), '--output', str(a.output.resolve()),
                 '--gpu', str(a.gpu), '--label', a.label, '--cases', a.cases]
        subprocess.run(child + ['--worker', 'record'], env=env, check=True)
        subprocess.run(child + ['--worker', 'render'], env=env, check=True)
        print('Video:', (a.output / 'rollout.mp4').resolve(), flush=True)


if __name__ == '__main__':
    main()
