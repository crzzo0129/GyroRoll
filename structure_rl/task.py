from copy import deepcopy
from functools import partial
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply,quat_from_euler_xyz

from miniwheelbot_rl.task import make_env_cfg,heading_rate
from miniwheelbot_rl.settings import Settings
from .model import MotorSettings,make_spec,MORPHOLOGIES,PHASES,controlled_joints
from .actions import StructureActionCfg


def joint_ids(env,morphology):
    names=controlled_joints(morphology)
    return env.scene['robot'].find_joints(names,preserve_order=True)[0]


def motor_state(env,morphology,settings):
    r=env.scene['robot'];ids=joint_ids(env,morphology)
    v=r.data.joint_vel[:,ids]
    if morphology=='twin_sync':
        return v/torch.tensor([settings.drive_limit,settings.rotor_limit,settings.rotor_limit],device=env.device)
    if morphology=='roll_linkage':
        passive=r.find_joints(('wheel_roll','passive_rod'),preserve_order=True)[0]
        # Retain passive positions and rates: the physical closed loop is compliant.
        return torch.cat((torch.stack((v[:,0]/settings.drive_limit,
            r.data.joint_pos[:,ids[1]]/settings.linkage_limit,v[:,1]/10.),dim=1),
            r.data.joint_pos[:,passive]/settings.linkage_limit,
            r.data.joint_vel[:,passive]/10.),dim=1)
    return torch.stack((v[:,0]/settings.drive_limit,r.data.joint_pos[:,ids[1]]/settings.steer_limit,v[:,1]/10.),dim=1)


def overspeed(env,morphology,settings):
    x=motor_state(env,morphology,settings)
    if morphology!='twin_sync':x=x[:,(0,2)]
    return (x.abs()-1).clamp_min(0).square().sum(dim=1)


def metric(env,name,morphology):
    r=env.scene['robot'];cmd=env.command_manager.get_command('twist')
    if name=='forward_mae':return (r.data.root_link_lin_vel_b[:,0]-cmd[:,0]).abs()
    if name=='yaw_mae':return (heading_rate(env)-cmd[:,2]).abs()
    if name=='roll_abs':
        g=r.data.projected_gravity_b
        return torch.atan2(-g[:,1],-g[:,2]).abs()
    if name=='closure_error':
        ids=r.find_sites(('rod_end','rocker_end'),preserve_order=True)[0]
        return torch.linalg.vector_norm(r.data.site_pos_w[:,ids[0]]-r.data.site_pos_w[:,ids[1]],dim=1)
    if name=='pitch_abs':
        g=r.data.projected_gravity_b
        return torch.atan2(g[:,0],torch.linalg.vector_norm(g[:,1:],dim=1)).abs()
    if name=='sync_error':
        ids=joint_ids(env,morphology)
        error=(r.data.joint_vel[:,ids[1]]-r.data.joint_vel[:,ids[2]]).abs()
        if not hasattr(env,'_sync_peak'):env._sync_peak=torch.zeros_like(error)
        torch.maximum(env._sync_peak,error,out=env._sync_peak)
        return error
    raise ValueError(name)


def reset_robot(env,env_ids,morphology,phase,profile='baseline',motion_fraction=1.,reset_tilt_deg=None):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    n=len(env_ids);r=env.scene['robot']
    state=r.data.default_root_state[env_ids].clone()
    state[:,:3]+=env.scene.env_origins[env_ids]
    state[:,2]+=.001
    angles=(torch.rand((n,3),device=env.device)*2-1)*torch.tensor([.015,.015,3.14159265],device=env.device)
    if profile in ('linkage_v2','linkage_v3','linkage_v4'):
        from .linkage_training import RESET_TILT
        angles[:,:2]*=RESET_TILT[phase]/.015
        if phase=='moving':
            angles[:,:2]*=(1/3+2/3*motion_fraction)
    if reset_tilt_deg is not None:
        angles[:,:2]=(torch.rand((n,2),device=env.device)*2-1)*(reset_tilt_deg*torch.pi/180)
    if getattr(env,'_fixed_reset',False):angles[:]=0
    if hasattr(env,'_evaluation_roll'):angles[:,0]=env._evaluation_roll[env_ids]
    if hasattr(env,'_evaluation_pitch'):angles[:,1]=env._evaluation_pitch[env_ids]
    q=quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2]);state[:,3:7]=q
    speed=torch.zeros(n,device=env.device)
    if profile in ('linkage_v2','linkage_v3','linkage_v4') and phase!='tracking':
        pass  # All early lessons start from rest, including low-speed starts.
    elif profile in ('steer_v2','linkage_v1','linkage_v2','linkage_v3','linkage_v4') and phase!='warmup':
        from .steer_training import sample_initial_speed
        speed=sample_initial_speed(n,env.device)
    elif phase=='moving':speed.uniform_(.5,1.)
    elif phase=='tracking':
        # Explicit mixture teaches starts/stops as well as moving balance.
        speed=torch.where(torch.rand(n,device=env.device)<.5,speed,torch.full_like(speed,.8))
    if hasattr(env,'_initial_speed'):speed[:]=env._initial_speed
    local_velocity=torch.zeros((n,3),device=env.device);local_velocity[:,0]=speed
    state[:,7:10]=quat_apply(q,local_velocity);state[:,10:13]=0
    pos=r.data.default_joint_pos[env_ids].clone();vel=torch.zeros_like(pos)
    ids=joint_ids(env,morphology)
    pos[:,ids[0]]=(torch.rand(n,device=env.device)*2-1)*3.14159265
    vel[:,ids[0]]=speed/.12
    if morphology=='twin_sync':
        # Identical positions and zero rotor speeds preserve the equality at reset.
        shared=(torch.rand(n,device=env.device)*2-1)*3.14159265
        pos[:,ids[1]]=shared;pos[:,ids[2]]=shared
    else:pos[:,ids[1]]=0
    if profile in ('linkage_v3','linkage_v4') and not getattr(env,'_airborne_reset',False):
        from .linkage_seed import grounded_height
        state[:,2]=env.scene.env_origins[env_ids,2]+grounded_height(angles[:,0],angles[:,1],pos[:,ids[0]])
    r.write_root_state_to_sim(state,env_ids=env_ids)
    r.write_joint_state_to_sim(pos,vel,env_ids=env_ids)


def yaw_error_cost(env):
    error=(heading_rate(env)-env.command_manager.get_command('twist')[:,2])/.3
    # Smooth L1-like tails continue distinguishing large errors when exp reward vanishes.
    return torch.sqrt(1+error.square())-1


def standing_reward(env):
    r=env.scene['robot'];cmd=env.command_manager.get_command('twist')
    standing=(cmd.abs().sum(dim=1)<1e-6)
    cost=(r.data.root_link_lin_vel_b[:,:2]/.12).square().sum(dim=1)+(heading_rate(env)/.15).square()
    return standing*(-r.data.projected_gravity_b[:,2]).clamp(0,1).square()*torch.exp(-cost)


def pitch_angle_cost(env):
    from .steer_training import pitch_angle_squared
    return pitch_angle_squared(env.scene['robot'].data.projected_gravity_b)


def linkage_posture(env):
    from .linkage_training import posture_reward
    return posture_reward(env.scene['robot'].data.projected_gravity_b)


def linkage_limit_cost(env,settings):
    from .linkage_training import limit_cost
    r=env.scene['robot'];ids=joint_ids(env,'roll_linkage')
    return limit_cost(r.data.joint_pos[:,ids[1]],settings.linkage_limit)


def alive_reward(env):
    return torch.ones(env.num_envs,device=env.device)


def configuration(morphology,phase,num_envs,settings=MotorSettings(),model_path=None,play=False,profile='baseline',motion_fraction=1.):
    if morphology not in MORPHOLOGIES or phase not in PHASES:raise ValueError((morphology,phase))
    if profile not in ('baseline','steer_v2','linkage_v1','linkage_v2','linkage_v3','linkage_v4'):raise ValueError(profile)
    if profile=='steer_v2' and (morphology!='raked_steer' or phase!='tracking'):
        raise ValueError('steer_v2 requires raked_steer tracking')
    if profile in ('linkage_v1','linkage_v2','linkage_v3','linkage_v4') and morphology!='roll_linkage':
        raise ValueError('Linkage profiles require roll_linkage')
    height=.42 if morphology=='roll_linkage' else .35
    physics=(Settings(initial_height=height,minimum_height=.26,timestep=.001,decimation=10)
             if morphology=='roll_linkage' else Settings(initial_height=height,minimum_height=.22))
    cfg=make_env_cfg(num_envs,'tracking',settings=physics,play=play)
    actuators=[XmlActuatorCfg(target_names_expr=('drive',),command_field='velocity')]
    if morphology=='twin_sync':actuators.append(XmlActuatorCfg(target_names_expr=('left','right'),command_field='velocity'))
    else:actuators.append(XmlActuatorCfg(target_names_expr=(controlled_joints(morphology)[1],),command_field='position'))
    cfg.scene.entities['robot']=EntityCfg(spec_fn=partial(make_spec,morphology,settings,model_path),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)),
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,height),joint_pos={'.*':0.},joint_vel={'.*':0.}))
    cfg.actions={'motors':StructureActionCfg(entity_name='robot',morphology=morphology,settings=settings)}
    terms=deepcopy(cfg.observations['actor'].terms)
    terms['wheel_velocity']=ObservationTermCfg(func=motor_state,params={'morphology':morphology,'settings':settings})
    cfg.observations={k:ObservationGroupCfg(terms=deepcopy(terms),enable_corruption=False) for k in ('actor','critic')}
    cfg.events={'reset_robot':EventTermCfg(func=reset_robot,mode='reset',params={'morphology':morphology,'phase':phase,'profile':profile,'motion_fraction':motion_fraction})}
    cfg.rewards['overspeed']=RewardTermCfg(func=overspeed,weight=-1.,params={'morphology':morphology,'settings':settings})
    cfg.rewards['upright'].weight=1.
    cmd=cfg.commands['twist']
    cmd.rel_standing_envs=0. if phase=='moving' else .25
    cmd.ranges.lin_vel_x=(.5,1.) if phase=='moving' else ((-.5,.5) if phase=='warmup' else (-1.,1.))
    yaw=.15 if phase=='warmup' else .3
    cmd.ranges.ang_vel_z=(-yaw,yaw)
    if profile in ('steer_v2','linkage_v1','linkage_v2','linkage_v3','linkage_v4'):
        cfg.rewards['yaw'].weight=4.
        cfg.rewards['pitch'].weight=-.05
        cfg.rewards['pitch_angle']=RewardTermCfg(func=pitch_angle_cost,weight=-.5)
        cfg.rewards['yaw_error']=RewardTermCfg(func=yaw_error_cost,weight=-.5)
        cfg.rewards['standing']=RewardTermCfg(func=standing_reward,weight=1.)
        if not play and phase=='tracking':
            from .steer_commands import BalancedSteeringCommandCfg
            cfg.commands['twist']=BalancedSteeringCommandCfg(entity_name='robot',
                ranges=deepcopy(cmd.ranges),resampling_time_range=(4.,8.),rel_standing_envs=.2)
    if profile in ('linkage_v2','linkage_v3','linkage_v4'):
        cfg.rewards['posture_precision']=RewardTermCfg(func=linkage_posture,weight=4.)
        cfg.rewards['alive']=RewardTermCfg(func=alive_reward,weight=2.)
        cfg.rewards['linkage_limit']=RewardTermCfg(func=linkage_limit_cost,weight=-2.,params={'settings':settings})
        cfg.rewards['failure'].weight=-100.
        cfg.rewards['action_rate'].weight=-.02
        if phase=='warmup':
            cmd.rel_standing_envs=1.
            cmd.ranges.lin_vel_x=(0.,0.)
            cmd.ranges.ang_vel_z=(0.,0.)
            cfg.rewards['forward'].weight=1.
            cfg.rewards['yaw'].weight=.5
        elif phase=='moving':
            cmd.rel_standing_envs=.25
            cmd.ranges.lin_vel_x=(-.4*motion_fraction,.4*motion_fraction)
            cmd.ranges.ang_vel_z=(0.,0.)
    if morphology=='roll_linkage':
        cfg.sim.mujoco.iterations=100
        cfg.sim.mujoco.tolerance=1e-10
    cfg.metrics={n:MetricsTermCfg(func=metric,params={'name':n,'morphology':morphology}) for n in ('forward_mae','yaw_mae','roll_abs')}
    if morphology=='roll_linkage':
        for name in ('closure_error','pitch_abs'):
            cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name,'morphology':morphology})
    if morphology=='twin_sync':cfg.metrics['sync_error']=MetricsTermCfg(func=metric,params={'name':'sync_error','morphology':morphology},per_substep=True)
    cfg.sim.nconmax=128;cfg.sim.njmax=512
    cfg.viewer.width=960;cfg.viewer.height=544;cfg.viewer.distance=1.3
    return cfg
