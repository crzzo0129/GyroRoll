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
from .model import MotorSettings,make_spec,MORPHOLOGIES,PHASES
from .actions import StructureActionCfg


def joint_ids(env,morphology):
    names=('drive','left','right') if morphology=='twin_sync' else ('drive','steer')
    return env.scene['robot'].find_joints(names,preserve_order=True)[0]


def motor_state(env,morphology,settings):
    r=env.scene['robot'];ids=joint_ids(env,morphology)
    v=r.data.joint_vel[:,ids]
    if morphology=='twin_sync':
        return v/torch.tensor([settings.drive_limit,settings.rotor_limit,settings.rotor_limit],device=env.device)
    return torch.stack((v[:,0]/settings.drive_limit,r.data.joint_pos[:,ids[1]]/settings.steer_limit,v[:,1]/10.),dim=1)


def overspeed(env,morphology,settings):
    x=motor_state(env,morphology,settings)
    if morphology=='raked_steer':x=x[:,(0,2)]
    return (x.abs()-1).clamp_min(0).square().sum(dim=1)


def metric(env,name,morphology):
    r=env.scene['robot'];cmd=env.command_manager.get_command('twist')
    if name=='forward_mae':return (r.data.root_link_lin_vel_b[:,0]-cmd[:,0]).abs()
    if name=='yaw_mae':return (heading_rate(env)-cmd[:,2]).abs()
    if name=='roll_abs':
        g=r.data.projected_gravity_b
        return torch.atan2(-g[:,1],-g[:,2]).abs()
    if name=='sync_error':
        ids=joint_ids(env,morphology)
        error=(r.data.joint_vel[:,ids[1]]-r.data.joint_vel[:,ids[2]]).abs()
        if not hasattr(env,'_sync_peak'):env._sync_peak=torch.zeros_like(error)
        torch.maximum(env._sync_peak,error,out=env._sync_peak)
        return error
    raise ValueError(name)


def reset_robot(env,env_ids,morphology,phase,profile='baseline'):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    n=len(env_ids);r=env.scene['robot']
    state=r.data.default_root_state[env_ids].clone()
    state[:,:3]+=env.scene.env_origins[env_ids]
    state[:,2]+=.001
    angles=(torch.rand((n,3),device=env.device)*2-1)*torch.tensor([.015,.015,3.14159265],device=env.device)
    if getattr(env,'_fixed_reset',False):angles[:]=0
    if hasattr(env,'_evaluation_roll'):angles[:,0]=env._evaluation_roll[env_ids]
    q=quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2]);state[:,3:7]=q
    speed=torch.zeros(n,device=env.device)
    if profile=='steer_v2':
        from .steer_training import sample_initial_speed
        speed=sample_initial_speed(n,env.device)
    elif phase=='moving':speed.uniform_(.5,1.)
    elif phase=='tracking':
        # Explicit mixture teaches starts/stops as well as moving balance.
        speed=torch.where(torch.rand(n,device=env.device)<.5,speed,torch.full_like(speed,.8))
    if hasattr(env,'_initial_speed'):speed[:]=env._initial_speed
    local_velocity=torch.zeros((n,3),device=env.device);local_velocity[:,0]=speed
    state[:,7:10]=quat_apply(q,local_velocity);state[:,10:13]=0
    r.write_root_state_to_sim(state,env_ids=env_ids)
    pos=r.data.default_joint_pos[env_ids].clone();vel=torch.zeros_like(pos)
    ids=joint_ids(env,morphology)
    pos[:,ids[0]]=(torch.rand(n,device=env.device)*2-1)*3.14159265
    vel[:,ids[0]]=speed/.12
    if morphology=='twin_sync':
        # Identical positions and zero rotor speeds preserve the equality at reset.
        shared=(torch.rand(n,device=env.device)*2-1)*3.14159265
        pos[:,ids[1]]=shared;pos[:,ids[2]]=shared
    else:pos[:,ids[1]]=0
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


def configuration(morphology,phase,num_envs,settings=MotorSettings(),model_path=None,play=False,profile='baseline'):
    if morphology not in MORPHOLOGIES or phase not in PHASES:raise ValueError((morphology,phase))
    if profile not in ('baseline','steer_v2'):raise ValueError(profile)
    if profile=='steer_v2' and (morphology!='raked_steer' or phase!='tracking'):
        raise ValueError('steer_v2 requires raked_steer tracking')
    cfg=make_env_cfg(num_envs,'tracking',settings=Settings(initial_height=.35,minimum_height=.22),play=play)
    actuators=[XmlActuatorCfg(target_names_expr=('drive',),command_field='velocity')]
    if morphology=='twin_sync':actuators.append(XmlActuatorCfg(target_names_expr=('left','right'),command_field='velocity'))
    else:actuators.append(XmlActuatorCfg(target_names_expr=('steer',),command_field='position'))
    cfg.scene.entities['robot']=EntityCfg(spec_fn=partial(make_spec,morphology,settings,model_path),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators)),
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,.35),joint_pos={'.*':0.},joint_vel={'.*':0.}))
    cfg.actions={'motors':StructureActionCfg(entity_name='robot',morphology=morphology,settings=settings)}
    terms=deepcopy(cfg.observations['actor'].terms)
    terms['wheel_velocity']=ObservationTermCfg(func=motor_state,params={'morphology':morphology,'settings':settings})
    cfg.observations={k:ObservationGroupCfg(terms=deepcopy(terms),enable_corruption=False) for k in ('actor','critic')}
    cfg.events={'reset_robot':EventTermCfg(func=reset_robot,mode='reset',params={'morphology':morphology,'phase':phase,'profile':profile})}
    cfg.rewards['overspeed']=RewardTermCfg(func=overspeed,weight=-1.,params={'morphology':morphology,'settings':settings})
    cfg.rewards['upright'].weight=1.
    cmd=cfg.commands['twist']
    cmd.rel_standing_envs=0. if phase=='moving' else .25
    cmd.ranges.lin_vel_x=(.5,1.) if phase=='moving' else ((-.5,.5) if phase=='warmup' else (-1.,1.))
    yaw=.15 if phase=='warmup' else .3
    cmd.ranges.ang_vel_z=(-yaw,yaw)
    if profile=='steer_v2':
        cfg.rewards['yaw'].weight=4.
        cfg.rewards['pitch'].weight=-.05
        cfg.rewards['pitch_angle']=RewardTermCfg(func=pitch_angle_cost,weight=-.5)
        cfg.rewards['yaw_error']=RewardTermCfg(func=yaw_error_cost,weight=-.5)
        cfg.rewards['standing']=RewardTermCfg(func=standing_reward,weight=1.)
        if not play:
            from .steer_commands import BalancedSteeringCommandCfg
            cfg.commands['twist']=BalancedSteeringCommandCfg(entity_name='robot',
                ranges=deepcopy(cmd.ranges),resampling_time_range=(4.,8.),rel_standing_envs=.2)
    cfg.metrics={n:MetricsTermCfg(func=metric,params={'name':n,'morphology':morphology}) for n in ('forward_mae','yaw_mae','roll_abs')}
    if morphology=='twin_sync':cfg.metrics['sync_error']=MetricsTermCfg(func=metric,params={'name':'sync_error','morphology':morphology},per_substep=True)
    cfg.sim.nconmax=128;cfg.sim.njmax=512
    cfg.viewer.width=960;cfg.viewer.height=544;cfg.viewer.distance=1.3
    return cfg
