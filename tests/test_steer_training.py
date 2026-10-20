"""Sampling coverage and physical sagittal-reflection checks."""
import unittest
import numpy as np
import mujoco
import torch
from structure_rl.model import model_xml
from structure_rl.steer_training import sample_commands, sample_initial_speed, mirror, OBS_REFLECTION, reset_exploration, pitch_angle_squared


class SteeringTrainingTests(unittest.TestCase):
    def test_commands_cover_stops_turns_and_mixed_speed_changes(self):
        torch.manual_seed(7)
        actual=torch.linspace(-1,1,100000)
        command,category=sample_commands(actual)
        expected=[.2,.2,.2,.15,.15,.1]
        for i,p in enumerate(expected):
            self.assertAlmostEqual((category==i).float().mean().item(),p,delta=.006)
        self.assertTrue((command[category==0]==0).all())
        self.assertTrue((command[category==1,0]>0).all())
        self.assertTrue((command[category==2,0]<0).all())
        self.assertTrue((command[category==3,2]>0).all())
        self.assertTrue((command[category==4,2]<0).all())
        for direction in (-1,1):
            changing=(category==5)&(actual*direction>.15)
            same_direction=(command[changing,0]*actual[changing]>0).float().mean().item()
            self.assertAlmostEqual(same_direction,.5,delta=.03)
        mixed=(category==5)&(actual>.3)
        self.assertTrue((command[mixed,0]>actual[mixed]).any())
        self.assertTrue(((command[mixed,0]>0)&(command[mixed,0]<actual[mixed])).any())
        mixed_commands=command[category==5]
        vx,yaw=mixed_commands[:,0],mixed_commands[:,2]
        for mask in ((vx>0)&(yaw==0),(vx<0)&(yaw==0),yaw>0,yaw<0):
            self.assertAlmostEqual(mask.float().mean().item(),.25,delta=.02)
        for turn in (yaw>0,yaw<0):
            self.assertAlmostEqual((vx[turn]>0).float().mean().item(),.5,delta=.04)
        self.assertTrue((command[:,1]==0).all())
        self.assertTrue((command[category!=0,0].abs()>=.3).all())
        self.assertLessEqual(command[:,0].abs().max(),1.)
        self.assertLessEqual(command[:,2].abs().max(),.3)

    def test_command_sampling_is_independent_of_actual_motion(self):
        torch.manual_seed(42)
        forward,_=sample_commands(torch.ones(10000))
        torch.manual_seed(42)
        reverse,_=sample_commands(-torch.ones(10000))
        torch.testing.assert_close(forward,reverse)

    def test_pitch_angle_cost_separates_pitch_from_roll(self):
        # Generate projected gravity from known roll/pitch combinations.
        roll=torch.tensor([0.,.5,-.5,.3,-.3])
        pitch=torch.tensor([0.,0.,0.,.4,-.4])
        gravity=torch.stack((pitch.sin(),-roll.sin()*pitch.cos(),-roll.cos()*pitch.cos()),dim=1)
        torch.testing.assert_close(pitch_angle_squared(gravity),pitch.square())

    def test_initial_speed_is_balanced(self):
        torch.manual_seed(8)
        speed=sample_initial_speed(100000,'cpu')
        for mask,p in ((speed==0,.5),(speed>0,.25),(speed<0,.25)):
            self.assertAlmostEqual(mask.float().mean().item(),p,delta=.006)
        self.assertTrue((speed.abs()<=.9).all())

    def test_mirror_is_involution_and_keeps_original_batch(self):
        from tensordict import TensorDict
        obs=TensorDict({k:torch.randn(5,16) for k in ('actor','critic')},batch_size=[5])
        actions=torch.randn(5,2)
        aug,act=mirror(obs,actions)
        torch.testing.assert_close(aug[:5],obs)
        torch.testing.assert_close(act[:5],actions)
        twice,act2=mirror(aug[5:],act[5:])
        torch.testing.assert_close(twice[5:],obs)
        torch.testing.assert_close(act2[5:],actions)
        torch.testing.assert_close(aug['actor'][5:],obs['actor']*torch.tensor(OBS_REFLECTION))

    def test_exploration_reset_restores_finite_std(self):
        from types import SimpleNamespace
        d=SimpleNamespace(min_std=torch.tensor(.02),max_std=torch.tensor(.2),
                          std_logit=torch.nn.Parameter(torch.full((2,),-20.)))
        reset_exploration(SimpleNamespace(distribution=d))
        torch.testing.assert_close(d.min_std+(d.max_std-d.min_std)*d.std_logit.sigmoid(),torch.full((2,),.08))

    def test_contact_dynamics_respect_left_right_reflection(self):
        model=mujoco.MjModel.from_xml_string(model_xml('raked_steer',ground=True))
        a,b=mujoco.MjData(model),mujoco.MjData(model)
        steer_q=model.joint('steer').qposadr[0]
        steer_v=model.joint('steer').dofadr[0]
        drive_v=model.joint('drive').dofadr[0]
        # quaternion reflection: (w,x,y,z) -> (w,-x,y,-z).
        qsign=np.ones(model.nq);qsign[[1,4,6,steer_q]]=-1
        vsign=np.ones(model.nv);vsign[[1,3,5,steer_v]]=-1
        a.qpos[2]=.349
        a.qpos[3:7]=[np.cos(.006),np.sin(.006),0,0]
        a.qpos[steer_q]=.04
        a.qvel[0]=.3;a.qvel[drive_v]=2.5
        b.qpos[:]=a.qpos*qsign;b.qvel[:]=a.qvel*vsign
        for step in range(150):
            a.ctrl[:]=[2.+np.sin(step*.03),.08*np.sin(step*.05)]
            b.ctrl[:]=a.ctrl*[1,-1]
            mujoco.mj_step(model,a);mujoco.mj_step(model,b)
            np.testing.assert_allclose(b.qpos,a.qpos*qsign,atol=2e-6,rtol=1e-5)
            np.testing.assert_allclose(b.qvel,a.qvel*vsign,atol=2e-5,rtol=1e-4)
        self.assertGreater(a.ncon,0)


if __name__=='__main__':
    unittest.main()
