from dataclasses import dataclass
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from .model import MotorSettings


@dataclass(kw_only=True)
class StructureActionCfg(ActionTermCfg):
    morphology: str
    settings: MotorSettings=MotorSettings()

    def build(self,env):return StructureAction(self,env)


class StructureAction(ActionTerm):
    """One drive action plus one shared rotor speed or one steering angle."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self._raw=torch.zeros((self.num_envs,2),device=self.device)
        self.applied=torch.zeros_like(self._raw)
        names=('drive','left','right') if cfg.morphology=='twin_sync' else ('drive','steer')
        ids,self.target_names=self._entity.find_joints(names,preserve_order=True)
        self.ids=torch.tensor(ids,device=self.device,dtype=torch.long)

    @property
    def action_dim(self):return 2

    @property
    def raw_action(self):return self._raw

    def process_actions(self,actions):
        self._raw[:]=actions
        self.applied[:]=actions.clamp(-1,1)

    def apply_actions(self):
        s=self.cfg.settings
        self._entity.set_joint_velocity_target(self.applied[:,:1]*s.drive_limit,joint_ids=self.ids[:1])
        if self.cfg.morphology=='twin_sync':
            shared=self.applied[:,1:2]*s.rotor_limit
            self._entity.set_joint_velocity_target(shared.expand(-1,2),joint_ids=self.ids[1:])
        else:
            self._entity.set_joint_position_target(self.applied[:,1:2]*s.steer_limit,joint_ids=self.ids[1:])

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self._raw[ids]=0.;self.applied[ids]=0.
