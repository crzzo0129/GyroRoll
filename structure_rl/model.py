"""Versioned morphology and actuator contracts, separate from the production bot."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
VERSION='structure_velocity_v1'
MORPHOLOGIES=('twin_sync','raked_steer')
PHASES=('warmup','moving','tracking')
# MuJoCo joint equalities are compliant. This is a validation tolerance,
# never an independent action or an allowed commanded speed difference.
SYNC_TOLERANCE_RAD_S=.01


@dataclass(frozen=True)
class MotorSettings:
    drive_limit: float=35.
    drive_kv: float=.25
    drive_torque: float=2.4
    rotor_limit: float=250.
    rotor_kv_each: float=.06
    rotor_torque_each: float=1.2
    steer_limit: float=.7
    steer_kp: float=8.
    steer_kd: float=.25
    steer_torque: float=2.4

    def __post_init__(self):
        if any(not 0<float(v)<float('inf') for v in asdict(self).values()):
            raise ValueError('Motor parameters must be finite and positive')


def source_model(morphology):
    if morphology not in MORPHOLOGIES:raise ValueError(morphology)
    return ROOT/'models'/f'{morphology}.xml'


def model_xml(morphology,settings=MotorSettings(),model_path=None,ground=False):
    if morphology not in MORPHOLOGIES:raise ValueError(morphology)
    root=ET.parse(model_path or source_model(morphology)).getroot()
    joints={j.get('name') for j in root.iter('joint') if j.get('axis')}
    expected={'drive','left','right'} if morphology=='twin_sync' else {'drive','steer'}
    if joints!=expected:raise ValueError(f'Unexpected joints: {joints}')
    if morphology=='twin_sync':
        eq=root.find("equality/joint[@name='rotor_ratio']")
        if eq is None or eq.get('joint1')!='left' or eq.get('joint2')!='right' or [float(x) for x in eq.get('polycoef','').split()]!=[0,1,0,0,0]:
            raise ValueError('Twin rotors require the fixed +1 equality constraint')
    world=root.find('worldbody')
    if not ground:
        option=root.find('option')
        if option is not None:root.remove(option)
        for node in list(world):
            if node.tag in ('geom','light'):world.remove(node)
    actuators=root.find('actuator');actuators.clear()
    def velocity(name,limit,kv,torque):
        ET.SubElement(actuators,'velocity',name=name+'_velocity',joint=name,kv=str(kv),gear='1',
            ctrllimited='true',ctrlrange=f'{-limit} {limit}',forcelimited='true',forcerange=f'{-torque} {torque}')
    velocity('drive',settings.drive_limit,settings.drive_kv,settings.drive_torque)
    if morphology=='twin_sync':
        for n in ('left','right'):velocity(n,settings.rotor_limit,settings.rotor_kv_each,settings.rotor_torque_each)
    else:
        ET.SubElement(actuators,'position',name='steer_position',joint='steer',kp=str(settings.steer_kp),kv=str(settings.steer_kd),gear='1',
            ctrllimited='true',ctrlrange=f'{-settings.steer_limit} {settings.steer_limit}',forcelimited='true',
            forcerange=f'{-settings.steer_torque} {settings.steer_torque}')
    return ET.tostring(root,encoding='unicode')


def make_spec(morphology,settings=MotorSettings(),model_path=None):
    import mujoco
    return mujoco.MjSpec.from_string(model_xml(morphology,settings,model_path))


def load_manifest(checkpoint,morphology=None):
    import json
    path=Path(checkpoint).parent
    manifest=json.loads((path/'task.json').read_text(encoding='utf-8'))
    if manifest.get('task_version')!=VERSION:raise ValueError('Checkpoint is not a structure_rl policy')
    if manifest.get('morphology') not in MORPHOLOGIES:raise ValueError('Unknown checkpoint morphology')
    if manifest.get('observation_dimension')!=16 or manifest.get('action_dimension')!=2:
        raise ValueError('Checkpoint observation/action contract mismatch')
    if morphology is not None and morphology!=manifest['morphology']:raise ValueError('Checkpoint morphology mismatch')
    if hashlib.sha256((path/'geometry.xml').read_bytes()).hexdigest()!=manifest['geometry_sha256']:
        raise ValueError('Checkpoint geometry hash mismatch')
    return manifest
