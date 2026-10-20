"""Independent concept comparison; finite-torque MuJoCo dynamics, no RL policy.

Uses a common simplified chassis and 240 x 24 mm wheel, not the production
geometry. LQR is identified for each morphology/speed; failures are controller
test failures, not proofs of mechanical impossibility.
"""
import argparse
from html import escape
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
try:
    from scipy.linalg import solve_discrete_are
    from scipy.optimize import brentq
except ImportError:
    # Keep the CPU probe usable in the minimal MuJoCo playback environment.
    def brentq(f, lo, hi, xtol=1e-13):
        flo=f(lo)
        if flo*f(hi)>0: raise ValueError('Unbracketed support height')
        while hi-lo>xtol:
            mid=(lo+hi)/2
            if f(mid)*flo>0: lo=mid
            else: hi=mid
        return (lo+hi)/2

    def solve_discrete_are(a,b,q,r):
        p=q.copy()
        for _ in range(20000):
            pn=q+a.T@p@a-a.T@p@b@np.linalg.solve(r+b.T@p@b,b.T@p@a)
            if np.max(abs(pn-p))<1e-8*max(1.,np.max(abs(pn))): return pn
            p=(pn+pn.T)/2
        raise ValueError('Riccati iteration did not converge')

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/comparisons/structure_concepts"
DT = .01


def model_output(kind):
    family = {'twin': 'twin_flywheel', 'steer': 'raked_steer'}[kind]
    return ROOT / 'results' / family / 'local_lqr'


def xml(kind, rake=30., trail=.03, coupling=None):
    if coupling not in (None,1,-1) or (coupling is not None and kind!='twin'):
        raise ValueError('coupling must be +1 or -1 for the twin-flywheel model')
    a = math.radians(rake)
    fork_x = trail + .12 * math.tan(a)
    common = '''<mujoco model="concept"><compiler angle="radian"/>
    <option timestep=".002" integrator="implicitfast" cone="elliptic" iterations="50"/>
    <default><joint damping=".00002" armature=".00002"/>
      <geom friction="1.6 .005 .0001" condim="3" solref=".008 1"/></default>
    <visual><global offwidth="960" offheight="640"/><headlight ambient=".5 .5 .5"/></visual>
    <asset><texture name="grid" type="2d" builtin="checker" width="128" height="128"
      rgb1=".8 .82 .85" rgb2=".65 .68 .72"/>
      <material name="floor" texture="grid" texrepeat="10 10"/></asset>
    <worldbody><light pos="0 -2 4"/><geom name="ground" type="plane" size="20 20 .1" material="floor"/>
    <body name="base" pos="0 0 .35"><freejoint name="root"/>
      <geom name="chassis" type="box" size=".055 .065 .10" mass=".645" rgba=".15 .45 .65 1"/>
      <geom name="tray" type="box" pos="0 0 .11" size=".08 .11 .005" mass="0" rgba=".8 .8 .8 1"/>
      <geom type="box" pos=".057 0 .06" size=".003 .04 .008" mass="0" rgba="1 .6 .1 1" contype="0" conaffinity="0"/>
    '''
    # Round 24 mm tire cross-section avoids a flat cylinder's artificial lateral
    # support plateau. Capsules approximate a torus; the wheel has explicit inertia.
    tread=[]
    for i in range(128):
        a0,a1=2*math.pi*i/128,2*math.pi*(i+1)/128
        tread.append(f'<geom name="tire_{i}" type="capsule" fromto="{.108*math.sin(a0)} 0 {.108*math.cos(a0)} {.108*math.sin(a1)} 0 {.108*math.cos(a1)}" size=".012" mass="0" rgba=".13 .14 .16 1"/>')
    wheel = '''<body name="drive_wheel" pos="{pos}"><joint name="drive" axis="0 1 0"/>
      <inertial pos="0 0 0" mass=".312" diaginertia=".00185 .00367 .00185"/>
      <geom type="cylinder" size=".10 .009" euler="1.57079632679 0 0" mass="0" rgba=".55 .58 .62 1" contype="0" conaffinity="0"/>
      '''+''.join(tread)+'''</body>'''
    if kind == "twin":
        parts = [common, '<geom name="support" type="capsule" fromto="0 0 -.1 0 0 -.23" size=".008" mass=".05" contype="0" conaffinity="0"/>', wheel.format(pos="0 0 -.23")]
        for name, side in (("left", 1), ("right", -1)):
            parts.append(f'''<body name="rotor_{name}" pos="0 {side*.085} -.23">
              <joint name="{name}" axis="1 0 0"/>
              <geom name="flywheel_{name}" type="cylinder" size=".055 .009" euler="0 1.57079632679 0" mass=".229" rgba=".9 .35 .15 1"/>
              <geom type="box" pos=".0095 .026 0" size=".001 .022 .002" mass="0" rgba="1 1 1 1" contype="0" conaffinity="0"/>
              </body>''')
        motors = ''.join(f'<motor name="{n}" joint="{n}" gear="1" ctrllimited="true" ctrlrange="-1.2 1.2"/>' for n in ('left','right'))
        contact = '<exclude body1="base" body2="drive_wheel"/>'
    else:
        parts = [common, f'''<body name="fork" pos="{fork_x} 0 -.23">
          <joint name="steer" axis="{math.sin(a)} 0 {math.cos(a)}" limited="true" range="-.7 .7"/>
          <geom name="fork_mass" type="capsule" fromto="0 0 .01 {-fork_x} 0 0" size=".009" mass=".05" contype="0" conaffinity="0"/>
          <geom name="axis_marker" type="capsule" fromto="{-math.sin(a)*.05} 0 {-math.cos(a)*.05} {math.sin(a)*.10} 0 {math.cos(a)*.10}" size=".006" mass="0" contype="0" conaffinity="0" rgba=".9 .35 .15 1"/>
          {wheel.format(pos=f'{-fork_x} 0 0')}</body>''']
        motors = '<motor name="steer" joint="steer" gear="1" ctrllimited="true" ctrlrange="-2.4 2.4"/>'
        contact = '<exclude body1="base" body2="drive_wheel"/>'
    equality='' if coupling is None else f'''<equality><joint name="rotor_ratio" joint1="left" joint2="right"
      polycoef="0 {coupling} 0 0 0" solref=".002 1" solimp=".9999 .9999 .001"/></equality>'''
    return ''.join(parts) + f'''</body></worldbody><contact>{contact}</contact>{equality}<actuator>
      <motor name="drive" joint="drive" gear="1" ctrllimited="true" ctrlrange="-2.4 2.4"/>
      {motors}</actuator></mujoco>'''


class Plant:
    def __init__(self, kind, rake=30., trail=.03, timestep=.002, coupling=None):
        self.kind = kind
        self.coupling=coupling
        self.xml = xml(kind, rake, trail,coupling)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.m.opt.timestep = timestep
        self.base = self.m.body('base').id
        self.names = ['drive','left','right'] if kind == 'twin' else ['drive','steer']
        self.dofs = [int(self.m.joint(n).dofadr[0]) for n in self.names]
        self.qids = [int(self.m.joint(n).qposadr[0]) for n in self.names]
        self.n = 8 if kind == 'twin' else 9

    def reset(self, speed=0., roll=0., pitch=0.):
        d = mujoco.MjData(self.m)
        d.qvel[0] = speed
        d.qvel[self.dofs[0]] = speed/.12
        def vertical(z):
            d.qpos[2] = z
            mujoco.mj_forward(self.m,d)
            return d.qacc[2]
        d.qpos[2] = brentq(vertical,.345,.351,xtol=1e-13)
        # Initial tilt is a stated perturbation, not a reset during the trial.
        qx = np.array([math.cos(roll/2),math.sin(roll/2),0.,0.])
        qy = np.array([math.cos(pitch/2),0.,math.sin(pitch/2),0.])
        mujoco.mju_mulQuat(d.qpos[3:7], qy, qx)
        mujoco.mj_forward(self.m,d)
        return d

    def state(self,d):
        r=d.xmat[self.base].reshape(3,3)
        roll=math.atan2(r[2,1],r[2,2])
        pitch=math.atan2(-r[2,0],math.hypot(r[2,1],r[2,2]))
        w=r.T@d.cvel[self.base,:3]
        v=r.T@d.qvel[:3]
        forward=r[:,0]; fdot=np.cross(d.cvel[self.base,:3],forward)
        heading_rate=(forward[0]*fdot[1]-forward[1]*fdot[0])/max(1e-8,float(forward[:2]@forward[:2]))
        x=[roll,pitch,w[0],w[1],v[0],v[1],heading_rate]
        if self.kind=='twin':
            sign=self.coupling if self.coupling is not None else 1
            x += [float((d.qvel[self.dofs[1]]+sign*d.qvel[self.dofs[2]])/2)]
        else:
            x += [float(d.qpos[self.qids[1]]),float(d.qvel[self.dofs[1]])]
        return np.array(x)

    def input(self,d,u):
        u=np.clip(u,-2.4,2.4)
        sign=self.coupling if self.coupling is not None else 1
        d.ctrl[:]=[u[0],u[1]/2,sign*u[1]/2] if self.kind=='twin' else u
        if self.kind=='twin':
            # Preserve the historical independent model for reproducing old results.
            # Coupled variants rely on the physical equality constraint instead.
            if self.coupling is None:
                diff=.001*(d.qvel[self.dofs[1]]-d.qvel[self.dofs[2]])
                d.ctrl[1:]+= [-diff,diff]
            d.ctrl[1:]=np.clip(d.ctrl[1:],-1.2,1.2)
            for i in (1,2):
                if abs(d.qvel[self.dofs[i]])>=250 and d.ctrl[i]*d.qvel[self.dofs[i]]>0:
                    d.ctrl[i]=0
        if abs(d.qvel[self.dofs[0]])>=35 and d.ctrl[0]*d.qvel[self.dofs[0]]>0:
            d.ctrl[0]=0

    def linearize(self,speed):
        trim=self.reset(speed)
        def advance(j=None,eps=0.,u=None):
            d=mujoco.MjData(self.m)
            d.qpos[:]=trim.qpos
            d.qvel[:]=trim.qvel
            if j in (0,1):
                q=np.zeros(4); axis=np.eye(3)[j]
                mujoco.mju_axisAngle2Quat(q,axis,eps)
                d.qpos[3:7]=q
            elif j in (2,3): d.qvel[j+1]+=eps
            elif j==4:
                d.qvel[0]+=eps
                d.qvel[self.dofs[0]]+=eps/.12
            elif j==5: d.qvel[1]+=eps
            elif j==6: d.qvel[5]+=eps
            elif j==7:
                if self.kind=='twin':
                    d.qvel[self.dofs[1]]+=eps
                    d.qvel[self.dofs[2]]+=eps*(self.coupling if self.coupling is not None else 1)
                else: d.qpos[self.qids[1]]+=eps
            elif j==8: d.qvel[self.dofs[1]]+=eps
            mujoco.mj_forward(self.m,d)
            self.input(d,np.zeros(2) if u is None else u)
            mujoco.mj_step(self.m,d,nstep=round(DT/self.m.opt.timestep))
            mujoco.mj_forward(self.m,d)
            return self.state(d)
        eps=1e-5
        a=np.column_stack([(advance(j,eps)-advance(j,-eps))/(2*eps) for j in range(self.n)])
        b=np.column_stack([(advance(u=np.eye(2)[j]*eps)-advance(u=-np.eye(2)[j]*eps))/(2*eps) for j in range(2)])
        return a,b,self.state(trim)


class Feedback:
    def __init__(self,plant,speed,effort=.2,integral_q=0.):
        self.p=plant
        self.a,self.b,self.trim=plant.linearize(speed)
        q=np.array([100.,100.,2.,2.,15.,3.,10.]+([.0005] if plant.kind=='twin' else [2.,.05]))
        # Small discount regularizes neutral/uncontrolled modes; report actual eigenvalues.
        discount=.999
        a,b=self.a,self.b
        if integral_q:
            a=np.eye(plant.n+1);a[:-1,:-1]=self.a;a[-1,6]=DT
            b=np.vstack((self.b,np.zeros((1,2))))
            q=np.r_[q,integral_q]
        p=solve_discrete_are(a*math.sqrt(discount),b*math.sqrt(discount),np.diag(q),np.eye(2)*effort)
        gain=np.linalg.solve(effort*np.eye(2)+discount*b.T@p@b,discount*b.T@p@a)
        self.k=gain[:,:plant.n]
        self.ki=gain[:,-1] if integral_q else np.zeros(2)
        self.integral=0.
        self.radius=float(max(abs(np.linalg.eigvals(a-b@gain))))

    def act(self,x,speed,turn):
        ref=self.trim.copy(); ref[4]=speed
        # A constant-speed turn is not a straight-trim equilibrium: use lean
        # feedforward and yaw feedback, never an unbounded least-squares trim.
        ref[0]=-speed*turn/9.81
        ref[6]=turn
        self.integral=np.clip(self.integral+(x[6]-turn)*DT,-1.,1.)
        return -self.k@(x-ref)-self.ki*self.integral


class ScheduledFeedback:
    def __init__(self,plant):
        self.fs=[Feedback(plant,v) for v in (0.,.25,.5,.75,1.)]
    def reset(self):
        for f in self.fs: f.integral=0.
    def act(self,x,speed,turn):
        index=np.clip(x[4],0.,1.)*4
        lo=min(3,int(index));alpha=index-lo
        return (1-alpha)*self.fs[lo].act(x,speed,turn)+alpha*self.fs[lo+1].act(x,speed,turn)


def trial(plant,fb,name,speed=1.,turn=0.,duration=15.,seed=0,record=False,heading_hold=False):
    rng=np.random.default_rng(seed)
    roll=rng.choice([-1.,1.])*math.radians(1.)
    fb.integral=0.
    if hasattr(fb,'reset'): fb.reset()
    d=plant.reset(speed if name.startswith('moving') else 0.,roll=roll)
    rows=[]; frames=[]; failed=False; steer_peak=0.; penetration=0.
    heading_ref=0.; last_heading=0.; yaw=0.; smooth_speed=0. if not name.startswith('moving') else speed
    for step in range(round(duration/DT)):
        t=step*DT
        x=plant.state(d)
        r=d.xmat[plant.base].reshape(3,3)
        current=math.atan2(r[1,0],r[0,0])
        yaw+=math.atan2(math.sin(current-last_heading),math.cos(current-last_heading));last_heading=current
        target_speed=0. if name=='stop' else speed
        if name=='stop_go': target_speed=0. if t<2 or t>=9 else speed
        smooth_speed+=np.clip(target_speed-smooth_speed,-.5*DT,.5*DT)
        target_turn=turn if t>=4 else 0.
        heading_ref+=target_turn*DT
        requested=target_turn+(np.clip(.8*(heading_ref-yaw),-.3,.3) if heading_hold else 0.)
        u=fb.act(x,smooth_speed,requested)
        plant.input(d,u)
        if plant.kind=='steer': steer_peak=max(steer_peak,abs(float(d.qpos[plant.qids[1]])))
        for contact in d.contact:
            penetration=max(penetration,-float(contact.dist))
        d.xfrc_applied[:]=0
        if name.endswith('push') and 6<=t<6.2: d.xfrc_applied[plant.base,1]=1.5
        rows.append([t,*x[:7],yaw,*d.qpos[:2],*d.ctrl, *d.qvel[plant.dofs]])
        if record and step%3==0: frames.append(d.qpos.copy())
        if abs(x[0])>.7 or abs(x[1])>.7 or d.qpos[2]<.22 or not np.isfinite(x).all():
            failed=True; break
        mujoco.mj_step(plant.m,d,nstep=round(DT/plant.m.opt.timestep))
        mujoco.mj_forward(plant.m,d)
    z=np.array(rows); mask=z[:,0]>3.; turnmask=z[:,0]>5.
    def rms(v): return float(np.sqrt(np.mean(v*v))) if len(v) else None
    metrics=dict(case=name,seed=seed,command_speed=0. if name=='stop' else speed,command_turn=turn,heading_hold=heading_hold,
        survived=not failed,duration_s=float(d.time),initial_roll_deg=math.degrees(roll),
        initial_speed_m_s=speed if name.startswith('moving') else 0.,timestep=float(plant.m.opt.timestep),
        forward_rmse=rms(z[mask,5]-(0. if name=='stop' else speed)) if name!='stop_go' else None,
        yaw_rate_rmse=rms(z[turnmask,7]-turn),heading_final_deg=math.degrees(yaw),
        heading_error_final_deg=math.degrees(yaw-heading_ref),lateral_final_m=float(d.qpos[1]),
        max_roll_deg=float(np.degrees(abs(z[:,1])).max()),
        max_pitch_deg=float(np.degrees(abs(z[:,2])).max()),max_penetration_mm=penetration*1000,
        planar_displacement_m=float(np.linalg.norm(d.qpos[:2])),
        torque_delta_rms=rms(np.diff(z[mask,11:11+plant.m.nu],axis=0)),
        peak_motor_speed=np.max(abs(z[:,11+plant.m.nu:]),axis=0).tolist(),
        saturation_fraction=float(np.mean(abs(z[:,11:11+plant.m.nu])>=np.array([2.4,1.2,1.2] if plant.kind=='twin' else [2.4,2.4])*.99)))
    if plant.kind=='steer': metrics['max_steer_deg']=math.degrees(steer_peak)
    metrics['tracking_pass']=bool(not failed and metrics['forward_rmse'] is not None and metrics['forward_rmse']<.15 and metrics['yaw_rate_rmse'] is not None and metrics['yaw_rate_rmse']<.10)
    if name in ('straight','moving_straight','moving_push'):
        metrics['heading_pass']=bool(not failed and abs(metrics['heading_error_final_deg'])<5.)
    return metrics,z,frames


def screen():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for kind,rake,trail in [('twin',0.,0.)]+[('steer',r,t) for r in (15.,30.,45.) for t in (-.16,-.12,-.06,0.,.06)]:
        plant=Plant(kind,rake,trail)
        item=dict(kind=kind,rake=rake,trail=trail,mass=float(plant.m.body_mass.sum()))
        try:
            fb=Feedback(plant,1.,integral_q=10. if kind=='steer' else 0.)
            cases=[trial(plant,fb,n,turn=turn,duration=10.)[0] for n,turn in [('moving_straight',0.),('moving_left',.2),('moving_right',-.2)]]
            cases.append(trial(plant,Feedback(plant,0.),'stop',duration=10.)[0])
            item.update(radius=fb.radius,cases=cases)
        except Exception as e: item['error']=str(e)
        results.append(item)
        print(json.dumps(item),flush=True)
        (OUT/'screen.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


CASES=[('stop',0.),('straight',0.),('moving_straight',0.),('moving_left',.2),
       ('moving_right',-.2),('moving_push',0.),('stop_go',0.)]
LABELS={'stop':'静止（零初速）','straight':'从零起步至 1 m/s','moving_straight':'直行（初速 1 m/s）',
        'moving_left':'左转 +0.2 rad/s（初速 1 m/s）','moving_right':'右转 −0.2 rad/s（初速 1 m/s）',
        'moving_push':'侧推 1.5 N × 0.2 s（初速 1 m/s）','stop_go':'零速→起步→停车'}


def save_controller(controller):
    if hasattr(controller,'fs'): return {'schedule_speeds':[0,.25,.5,.75,1.], 'controllers':[save_controller(f) for f in controller.fs]}
    return {'trim':controller.trim.tolist(),'K':controller.k.tolist(),'Ki':controller.ki.tolist(),
            'linear_closed_loop_radius':controller.radius,'A':controller.a.tolist(),'B':controller.b.tolist()}


def validate():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'description':'Independent empty-load concept models; not production geometry or PPO checkpoints.',
            'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,
            'controller_dt_s':DT,'round_tire_segments':128,'rollout_seconds':15,
            'assumptions':{'wheel_diameter_m':.24,'wheel_width_m':.024,'chassis_mass_kg':.695,
              'twin_rotor_radius_m':.055,'twin_each_rotor_mass_kg':.229,'twin_each_peak_torque_Nm':1.2,
              'drive_peak_torque_Nm':2.4,'steer_peak_torque_Nm':2.4,'rotor_speed_cutoff_rad_s':250,
              'drive_speed_cutoff_rad_s':35,'steer_limit_rad':.7,'rake_deg':15.,'ground_trail_m':-.16,
              'actuation':'100 Hz clipped torque feedback; cutoff prevents accelerating further beyond speed bound, not a hard speed clamp',
              'tire_friction':1.6,'random_mass_or_friction':False,'initial_roll_deg':[-1,1],
              'initial_flywheel_speed':0.,'payload':None,'fixed_chassis_geometry':True},'models':{},'cases':[]}
    for kind in ('twin','steer'):
        plant=Plant(kind,15.,-.16)
        folder=model_output(kind);folder.mkdir(parents=True,exist_ok=True)
        (folder/'model.xml').write_text(plant.xml,encoding='utf-8')
        schedule=ScheduledFeedback(plant)
        moving= schedule if kind=='twin' else Feedback(plant,1.,integral_q=10.)
        stop= schedule if kind=='twin' else Feedback(plant,0.,integral_q=10.)
        controls={'moving':save_controller(moving),'stop':save_controller(stop),'startup':save_controller(schedule)}
        (folder/'controller.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')
        d=plant.reset()
        report['models'][kind]={'mass_kg':float(plant.m.body_mass.sum()),'upright_com_height_m':float(d.subtree_com[plant.base,2]),
            'geometry_sha256':hashlib.sha256((folder/'model.xml').read_bytes()).hexdigest(),
            'motors':plant.names,'action_count':plant.m.nu}
        if kind=='twin':
            report['models'][kind]['rotor_spin_inertia_total']=sum(float(plant.m.body('rotor_'+n).inertia.max()) for n in ('left','right'))
        for dt in (.002,.001,.0005):
            plant.m.opt.timestep=dt
            for seed in (0,1):
                for name,turn in CASES:
                    fb=stop if name=='stop' else schedule if name in ('straight','stop_go') else moving
                    record=dt==.002 and seed==0
                    metrics,z,frames=trial(plant,fb,name,turn=turn,seed=seed,record=record)
                    metrics['kind']=kind
                    report['cases'].append(metrics)
                    if record:
                        header=['time_s','roll_rad','pitch_rad','roll_rate_rad_s','pitch_rate_rad_s','forward_m_s','lateral_m_s','heading_rate_rad_s','heading_rad','x_m','y_m']
                        header += [n+'_torque_Nm' for n in plant.names]+[n+'_speed_rad_s' for n in plant.names]
                        np.savetxt(folder/(name+'.csv'),z,delimiter=',',header=','.join(header),comments='')
                        np.savez_compressed(folder/(name+'.npz'),qpos=np.array(frames),frame_dt=3*DT)
                    print(kind,dt,seed,name,'survived',metrics['survived'],'heading',round(metrics['heading_error_final_deg'],2),flush=True)
            (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def render():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    previews=[]
    for kind in ('twin','steer'):
        folder=model_output(kind)
        model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
        data=mujoco.MjData(model)
        camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
        camera.distance=1.0;camera.azimuth=135.;camera.elevation=-15.
        with mujoco.Renderer(model,height=640,width=960) as renderer:
            mujoco.mj_forward(model,data);camera.lookat[:]=[0,0,.23]
            renderer.update_scene(data,camera=camera)
            frame=renderer.render()
            imageio.imwrite(folder/'preview.png',frame)
            previews.append(Image.fromarray(frame))
            for case in ('straight','moving_left','moving_push','stop'):
                with np.load(folder/(case+'.npz')) as trajectory:
                    qs=trajectory['qpos'];frame_dt=float(trajectory['frame_dt'])
                with imageio.get_writer(str(folder/(case+'.mp4')),fps=1/frame_dt,codec='libx264',quality=7) as writer:
                    for q in qs:
                        data.qpos[:]=q;mujoco.mj_forward(model,data)
                        camera.lookat[:]=[data.qpos[0],data.qpos[1],.23]
                        renderer.update_scene(data,camera=camera)
                        writer.append_data(renderer.render())
                print('Rendered',kind,case,flush=True)
    canvas=Image.new('RGB',(1920,690),'white')
    for i,p in enumerate(previews):canvas.paste(p,(960*i,50))
    draw=ImageDraw.Draw(canvas);draw.text((25,15),'A: TWO PARALLEL ROLL FLYWHEELS',fill='black');draw.text((985,15),'B: RAKED STEERING, NO FLYWHEEL',fill='black')
    canvas.save(OUT/'structures.png')


def chart(title,z,columns,unit):
    colors=['#0072b2','#d55e00','#009e73']
    lo=min(float(z[:,j].min()) for _,j in columns);hi=max(float(z[:,j].max()) for _,j in columns)
    pad=max(.01,.1*(hi-lo));lo-=pad;hi+=pad
    paths=[]
    for i,(label,j) in enumerate(columns):
        points=' '.join(f'{55+690*t/15:.1f},{220-190*(v-lo)/(hi-lo):.1f}' for t,v in zip(z[:,0],z[:,j]))
        paths.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="1"/>')
    legend=' / '.join(f'<span style="color:{colors[i]}">{escape(label)}</span>' for i,(label,_) in enumerate(columns))
    return f'<section><h3>{title}</h3><svg viewBox="0 0 780 250"><path d="M55 25V220H745" fill="none" stroke="#aaa"/><text x="0" y="35">{hi:.2f}</text><text x="0" y="220">{lo:.2f}</text><text x="55" y="242">0 s</text><text x="710" y="242">15 s</text>{"".join(paths)}</svg><p>{legend} · {unit}</p></section>'


def write_report():
    report=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    rows=[]
    for kind,label in [('twin','双飞轮'),('steer','斜轴转向')]:
        for name,_ in CASES:
            cases=[c for c in report['cases'] if c['kind']==kind and c['case']==name]
            nominal=next(c for c in cases if c['seed']==0 and c['timestep']==.002)
            def fmt(k):return '—' if nominal.get(k) is None else f'{nominal[k]:.3f}'
            passed='—' if name=='stop_go' else f'{sum(c["tracking_pass"] for c in cases)}/6'
            rows.append(f'<tr><td>{label}</td><td>{LABELS[name]}</td><td>{sum(c["survived"] for c in cases)}/6</td><td>{passed}</td><td>{fmt("forward_rmse")}</td><td>{fmt("yaw_rate_rmse")}</td><td>{fmt("heading_error_final_deg")}</td></tr>')
    plots=[]
    for kind,label in [('twin','双飞轮'),('steer','斜轴转向')]:
        for name in ('straight','moving_straight','moving_left','moving_push'):
            z=np.genfromtxt(model_output(kind)/(name+'.csv'),delimiter=',',skip_header=1)
            plots.append(chart(label+' · '+LABELS[name],z,[('横滚角',1),('航向角',8)],'rad'))
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>两种结构重新验证</title><style>body{max-width:1250px;margin:35px auto;padding:0 22px;font-family:system-ui,sans-serif;color:#17283c;background:#f5f7fa;line-height:1.7}h1{font-size:30px}section,.card{background:white;padding:18px;border-radius:10px;margin:16px 0}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img,video{max-width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}svg{width:100%;font-size:12px}.notice{border-left:5px solid #cb6f16;padding:14px;background:#fff4e5}@media(max-width:800px){.grid{grid-template-columns:1fr}table{font-size:12px}}</style>
    <h1>双小飞轮 vs 斜轴转向：本地动力学验证</h1><p>2026-10-19 · 独立概念模型 · MuJoCo CPU · 有限力矩反馈控制</p>
    <p class="notice">双飞轮的静止/起步可行性已在这些测试中得到支持，但没有解决航向控制；斜轴转向在已有前进速度时更能控制航向，零速平衡与起步尚未通过。测试失败不代表结构在所有控制策略下不可能成功。</p>
    <img src="structures.png" alt="双飞轮与斜轴转向模型对照">
    <section><h2>模型与公平性边界</h2><p>两种方案采用同一简化机身、直径 240 mm / 宽 24 mm 的圆截面滚轮。双飞轮半径各 55 mm、质量各 229 g、轴向均为前后 x 轴，位于滚轮左右 y=±85 mm、轴心高 120 mm；无预旋转。各飞轮峰值力矩 1.2 N·m，总计 2.4 N·m。滚轮电机峰值为 2.4 N·m。</p>
    <p>斜轴方案完全去掉飞轮，转向轴在前后平面相对竖直倾斜 15°，地面拖距 −160 mm，转向范围 ±0.7 rad，转向力矩上限 2.4 N·m。拖距定义为转向轴与地面的交点相对滚轮中心的前后距离；负值表示轴线交点在后方。这是筛选出的概念布局，长负拖距尚未做机械包装设计。</p>
    <p>双飞轮总质量 1.465 kg，斜轴方案 1.007 kg；去掉飞轮后没有加等质量配重，所以结果比较的是两种实际质量预算，不能把差异全归因于转向机构。机身、支架、电机质量均为概念预算。模型没有货物，不是当前 PPO checkpoint 的几何。</p>
    <p>两个平行飞轮的机身直接反作用力矩为 −(τ左+τ右)·x轴。左右位置不使它们变成地面差速轮；两电机的差动不能独立提供直接 yaw 力矩。两个小飞轮的总轴向惯量约 0.000693 kg·m²，小于原飞轮约 0.001520 kg·m²，相同角动量需求会需要更高转速。</p></section>
    <section><h2>方法</h2><p>每个工况测试初始 roll ±1°，在 2 / 1 / 0.5 ms 物理步长各回放一次，共 6 次；控制周期均为 10 ms，细步长沿用 2 ms 识别的控制器。无运行中状态重置、无隐藏支撑、无初始飞轮预旋转。行进组明确给定 1 m/s 初速度及对应滚轮速度，不能用它证明自主起步。</p>
    <p>双飞轮使用 0–1 m/s 的局部 LQR 增益调度；斜轴行进使用 1 m/s 的 LQR 加航向角速度误差积分。静止使用单独的零速控制器，起步用增益调度。力矩直接限幅，不是生产 RL 的速度目标伺服，不能直接比较 action_rate。驱动轮/飞轮超出 35/250 rad/s 时禁止继续同向加速，但速度可因惯性或外力越界。</p>
    <p>速度命令按 0.5 m/s² 过渡，转向在 4 s 加入 ±0.2 rad/s；侧推在 6–6.2 s 对机身施加世界 +y 方向 1.5 N 外力。起停工况为 0–2 s 停止、2–9 s 行进、9–15 s 停车。roll/pitch 超过 0.7 rad 或根节点低于 0.22 m 判失败并结束。</p></section>
    <h2>结果</h2><p>不倒与跟踪分别计数。跟踪通过要求完整存活，3 s 后速度 RMSE &lt;0.15 m/s、5 s 后航向角速度 RMSE &lt;0.10 rad/s。这个判据只约束速度和角速度，不保证累计航向正确，必须同时看末端航向误差。起停工况只统计存活，不纳入这个固定命令跟踪判据。表中误差为 +1° 初始 roll、2 ms 步长的代表回放；完整 84 条结果在 summary.json。失败回放的短期误差不能当作通过。</p>
    <table><tr><th>结构</th><th>工况</th><th>存活</th><th>跟踪通过</th><th>速度 RMSE m/s</th><th>航向角速度 RMSE rad/s</th><th>末端航向误差 °</th></tr>'''+''.join(rows)+'''</table>
    <section><h2>观看代表回放</h2><p>失败视频在失败点停止，不拼接重置后的画面。</p><div class="grid">
    <div><h3>双飞轮：从零起步</h3><video controls src="../../twin_flywheel/local_lqr/straight.mp4"></video></div>
    <div><h3>斜轴转向：已有初速下左转</h3><video controls src="../../raked_steer/local_lqr/moving_left.mp4"></video></div>
    <div><h3>双飞轮：行进侧推</h3><video controls src="../../twin_flywheel/local_lqr/moving_push.mp4"></video></div>
    <div><h3>斜轴转向：行进侧推</h3><video controls src="../../raked_steer/local_lqr/moving_push.mp4"></video></div></div></section>
    <h2>姿态与航向</h2><div class="grid">'''+''.join(plots)+'''</div>
    <section><h2>限制与下一步</h2><p>这是两个新概念模型的局部可行性测试，未训练新 PPO、未复用旧结构策略。当前结果尚不足以决定最终硬件：双飞轮未表现出独立航向控制优势；斜轴方案需要解决低速起停，并缩短负拖距或优化机身/轴线后重新验证。抗推“未倒”不等于恢复原路径，尤其不能作为运输光滑货物已通过的证据。</p><p>本报告没有实测电机转矩—转速曲线、延迟、传感噪声、货物滑动或地面随机化。轮胎为 128 段胶囊构成的圆截面近似；细步长一致性不等于接触模型或硬件真实性已验证。</p><p>复现：在项目根目录运行 <code>.venv-mjlab/Scripts/python.exe probe_structures.py --validate --render</code>。目录保留两个 model.xml、controller.json、CSV、逐帧 qpos、MP4 和总表。</p></section></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.screen:screen()
    if args.validate:validate()
    if args.render:render()
    if args.validate or args.render or args.report:write_report()
