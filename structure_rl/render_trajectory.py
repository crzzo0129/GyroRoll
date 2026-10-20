"""Render recorded GPU trajectories in a separate EGL process; no physics stepping."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--stage',type=int,required=True)
    p.add_argument('--input',type=Path,default=Path('results/steer_v2_trajectories'))
    p.add_argument('--output',type=Path,default=Path('results/steer_v2_final'));a=p.parse_args()
    stem=f'{a.name}_stage{a.stage}';trajectory=np.load(a.input/(stem+'.npz'))
    report=json.loads((a.input/(stem+'.json')).read_text())
    model=mujoco.MjModel.from_xml_path(str(a.input/'scene/scene.xml'))
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
                case,t,target,measured,yaw,roll,failed,failed_time=meta
                data.qpos[:]=qpos
                mujoco.mj_forward(model,data)  # Forward kinematics only, never mj_step.
                camera.lookat[:]=data.xpos[base_id]+np.array([0.,0.,-.06])
                renderer.update_scene(data,camera=camera)
                frame=Image.new('RGB',(640,448),(15,23,36));frame.paste(Image.fromarray(renderer.render()),(0,0))
                draw=ImageDraw.Draw(frame)
                label=('BALANCED' if a.name=='balanced' else 'BALANCED + MIRROR')+f' | +{a.stage*2000} updates'
                draw.rectangle((0,0,640,30),fill=(15,23,36));draw.text((12,5),label,font=font,fill='white')
                title=report['cases'][int(case)]['name']
                draw.text((12,365),f'{title} | t={t:04.1f}s | target v={target:+.1f} m/s',font=font,fill=(120,220,255))
                draw.text((12,390),f'actual v={measured:+.2f} m/s | yaw rate={yaw:+.2f} rad/s | roll={roll:+.1f} deg',font=font,fill='white')
                state=(f'TERMINATED at {failed_time:.2f}s - last pose held' if failed else
                       ('RESET: zero speed, zero roll; no action filtering' if t<2 else 'Continuous rollout; target yaw rate = 0'))
                draw.text((12,417),state,font=font,fill=(255,110,100) if failed else (190,200,215))
                writer.append_data(np.array(frame))
                if index in (0,300,650,950):frame.save(a.output/(stem+f'_frame{index}.png'))
                if index%240==0:print(stem,'frame',index,flush=True)
    report['rendering']='MuJoCo 3.10 EGL physical GPU 1; recorded qpos replay, no re-simulation'
    (a.output/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VIDEO_COMPLETE',stem,flush=True)


if __name__=='__main__':main()
