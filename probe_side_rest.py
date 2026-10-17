"""Compare natural side-rest tire contact before and after the lower-body cut."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe_rise import Experiment


def measurements(experiment, data):
    m = experiment.model
    ground, drive = m.geom('ground').id, m.body('drive_wheel').id
    tire_low = math.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != drive:
            continue
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices = m.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
        tire_low = min(tire_low, float(vertices[:,2].min()))
    names, wheel_load, total_load = [], 0., 0.
    for i, contact in enumerate(data.contact):
        if ground not in contact.geom:
            continue
        other = next(g for g in contact.geom if g != ground)
        force = np.zeros(6)
        mujoco.mj_contactForce(m, data, i, force)
        total_load += force[0]
        if m.geom_bodyid[other] == drive:
            wheel_load += force[0]
            names.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other))
    return dict(tilt_deg=math.degrees(experiment.state(data)[4]), tire_bottom_m=tire_low,
                tire_contact_geoms=names, tire_normal_force_N=wheel_load,
                tire_load_fraction=wheel_load/total_load, total_ground_normal_force_N=total_load)


def main():
    output = Path('results/chamfer_rise')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    canvas=Image.new('RGB',(1800,650),'#edf1f2')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    draw.text((36,20),'侧放接地对比 / 从 90° 放下，自然静置 2 秒',font=font,fill='#183740')
    for index,(name,path) in enumerate((('修改前','results/local_rise/geometry.xml'),('下部斜切后','wheelbot/nested_model.xml'))):
        for timestep in (.002,.001,.0005):
            ex=Experiment(timestep=timestep,model_path=path)
            for roll in (90.,-90.):
                d=ex.reset(roll=roll,settle=2.)
                result=measurements(ex,d)
                results.append(dict(layout=name,physics_timestep=timestep,initial_roll_deg=roll,**result))
                if index==1:
                    assert result['tire_normal_force_N']>1
                if timestep!=.002 or roll!=90.:
                    continue
                m=ex.model
                m.vis.global_.offwidth,m.vis.global_.offheight=900,600
                m.vis.quality.offsamples=0
                camera=mujoco.MjvCamera()
                camera.lookat[:]=(0,0,.10)
                camera.azimuth,camera.elevation,camera.distance=180,-8,1.10
                with mujoco.Renderer(m,height=600,width=900) as renderer:
                    renderer.update_scene(d,camera=camera)
                    canvas.paste(Image.fromarray(renderer.render().copy()).crop((0,130,900,560)),(900*index,90))
                draw.text((900*index+36,540),f"{name}：自然侧躺倾角 {result['tilt_deg']:.1f}°",font=font,fill='#183740')
                draw.text((900*index+36,584),f"轮胎离地 {max(0,result['tire_bottom_m'])*1000:.1f} mm  /  轮胎承重 {result['tire_normal_force_N']:.2f} N",font=small,fill='#183740')
    canvas.save(output/'side_rest_comparison.png')
    (output/'side_rest.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(results[:2]+results[6:8],indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
