import copy
import math
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch

from structure_rl.linkage_training import gate_cases,limit_cost,posture_reward,promotion_passed,candidate_acceptable
from structure_rl.linkage_seed import affine_seed,initialize_actor,grounded_height


class LinkageTrainingTests(unittest.TestCase):
    def test_direct_tracking_launches_one_shared_policy_on_two_workers(self):
        from structure_rl.linkage_curriculum import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);seed=root/'model_0.pt';seed.touch()
            seed.with_name('task.json').write_text(json.dumps({'training_profile':'linkage_v4'}))
            commands=[]
            def run(command,check):
                commands.append(command)
                if 'train' in command:
                    directory=Path(command[command.index('--run-dir')+1])
                    directory.mkdir(parents=True);(directory/'model_1.pt').touch()
                    return
                phase=command[command.index('--phase')+1]
                output=Path(command[command.index('--output')+1])
                output.parent.mkdir(parents=True,exist_ok=True)
                output.write_text(json.dumps({'phase':phase,'duration_s':15.,'trials_per_case':16,
                    'cases':[{'name':name,'survival_rate':1.,'pass_rate':1. if phase=='warmup' else .5,
                              'mean_survival_s':15.,'max_closure_error_m':.0001}
                             for name,_,_ in gate_cases(phase)]}))
            with patch('sys.argv',['curriculum','--output',str(root/'run'),'--profile','linkage_v4',
                    '--checkpoint',str(seed),'--max-blocks','1','--direct-tracking',
                    '--gpus','2','--num-envs','1024','--reset-tilt-deg','.1']), \
                    patch('structure_rl.linkage_curriculum.subprocess.run',side_effect=run):
                main()
            training=[c for c in commands if 'train' in c]
            self.assertEqual(len(training),1)
            command=training[0]
            self.assertIn('torch.distributed.run',command)
            self.assertIn('--nproc_per_node=2',command)
            self.assertEqual(command[command.index('--num-envs')+1],'512')
            self.assertEqual(command[command.index('--phase')+1],'tracking')
            self.assertEqual(command[command.index('--reset-tilt-deg')+1],'0.1')
            self.assertFalse(any('moving' in c for c in commands))
            tracking_checks=[c for c in commands if 'structure_rl.linkage_gate' in c and 'tracking' in c]
            self.assertTrue(all('--tilt-deg' in c for c in tracking_checks))

    def test_curriculum_resumes_preserved_checkpoint_after_regression(self):
        from structure_rl.linkage_curriculum import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);seed=root/'model_0.pt';seed.touch()
            seed.with_name('task.json').write_text(json.dumps({'training_profile':'linkage_v4'}))
            commands=[]
            def run(command,check):
                commands.append(command)
                if command[3]=='structure_rl':
                    directory=Path(command[command.index('--run-dir')+1])
                    directory.mkdir(parents=True)
                    (directory/'model_1.pt').touch()
                    return
                phase=command[command.index('--phase')+1]
                output=Path(command[command.index('--output')+1])
                output.parent.mkdir(parents=True,exist_ok=True)
                report={'phase':phase,'duration_s':12. if phase=='warmup' else 15.,
                    'trials_per_case':16,'cases':[
                        {'name':name,'survival_rate':1.,'pass_rate':1. if phase=='warmup' else .5,
                         'mean_survival_s':12. if phase=='warmup' else 15.,
                         'max_closure_error_m':.0001} for name,_,_ in gate_cases(phase)]}
                if 'block_00' in output.parts:
                    report['cases'][0]['survival_rate']=0.
                output.write_text(json.dumps(report))
            with patch('sys.argv',['curriculum','--output',str(root/'run'),'--profile','linkage_v4',
                    '--checkpoint',str(seed),'--max-blocks','2','--block-iterations','2']), \
                    patch('structure_rl.linkage_curriculum.subprocess.run',side_effect=run):
                main()
            training=[c for c in commands if c[3]=='structure_rl']
            self.assertEqual(len(training),2)
            for command in training:
                self.assertEqual(command[command.index('--checkpoint')+1],str(seed))
            self.assertEqual(float(training[1][training[1].index('--learning-rate')+1]),5e-7)
            status=json.loads((root/'run/status.json').read_text())
            self.assertEqual(status['status'],'needs_revision')
            self.assertFalse(status['history'][1]['accepted'])
            self.assertTrue(status['history'][2]['guards']['warmup'])

    def test_relu_seed_and_small_update_have_no_bias_cancellation(self):
        actor=torch.nn.Sequential(torch.nn.Linear(20,128),torch.nn.ReLU(),
            torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,128),
            torch.nn.ReLU(),torch.nn.Linear(128,2))
        initialize_actor(actor,activation='relu')
        torch.manual_seed(42)
        obs=torch.randn(128,20)*2
        weight,bias=affine_seed()
        torch.testing.assert_close(actor(obs),obs@weight.T+bias,atol=.0002,rtol=.0001)
        torch.testing.assert_close(actor[-1].bias,bias)
        actor(obs).square().mean().backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in actor.parameters()))

    def test_rollback_preserves_each_case_and_rejects_invalid_physics(self):
        baseline={'phase':'moving','duration_s':15.,'trials_per_case':16,
            'cases':[{'name':name,'survival_rate':1.,'pass_rate':.5,
                      'mean_survival_s':15.,'max_closure_error_m':.0001}
                     for name,_,_ in gate_cases('moving')]}
        self.assertTrue(candidate_acceptable(copy.deepcopy(baseline),baseline))
        for key,value in (('survival_rate',.9),('pass_rate',.4),
                          ('mean_survival_s',14.),('max_closure_error_m',.002),
                          ('mean_survival_s',float('nan'))):
            bad=copy.deepcopy(baseline);bad['cases'][1][key]=value
            self.assertFalse(candidate_acceptable(bad,baseline))
        good=copy.deepcopy(baseline);good['cases'][1]['pass_rate']=.75
        self.assertTrue(candidate_acceptable(good,baseline))
        bad=copy.deepcopy(good);bad['controller']='affine_reference'
        self.assertFalse(candidate_acceptable(bad,baseline))

    def test_seed_is_a_trainable_mlp_with_expected_feedback(self):
        actor=torch.nn.Sequential(torch.nn.Linear(20,128),torch.nn.ELU(),
            torch.nn.Linear(128,128),torch.nn.ELU(),torch.nn.Linear(128,128),
            torch.nn.ELU(),torch.nn.Linear(128,2))
        initialize_actor(actor)
        torch.manual_seed(42)
        obs=torch.randn(128,20)*.15
        obs[:,8]=-1.
        weight,bias=affine_seed()
        torch.testing.assert_close(actor(obs),obs@weight.T+bias,atol=.0004,rtol=.0001)
        self.assertTrue(all(p.requires_grad for p in actor.parameters()))
        self.assertNotIn('gain',dict(actor.named_buffers()))

    def test_grounded_reset_matches_tire_geometry(self):
        import mujoco
        import numpy as np
        from structure_rl.model import model_xml
        model=mujoco.MjModel.from_xml_string(model_xml('roll_linkage',ground=True))
        data=mujoco.MjData(model)
        tire_ids=[model.geom(f'tire_{i}').id for i in range(128)]
        for roll,pitch,phase in ((0.,0.,0.),(.01,-.02,.012),(-.02,.01,1.2)):
            data.qpos[3:7]=[1,0,0,0]
            qx=np.array([math.cos(roll/2),math.sin(roll/2),0,0])
            qy=np.array([math.cos(pitch/2),0,math.sin(pitch/2),0])
            mujoco.mju_mulQuat(data.qpos[3:7],qy,qx)
            data.qpos[model.joint('drive').qposadr[0]]=phase
            data.qpos[2]=grounded_height(*(torch.tensor([v]) for v in (roll,pitch,phase))).item()
            mujoco.mj_forward(model,data)
            axis=data.geom_xmat[tire_ids].reshape(-1,3,3)[:,2,2]
            bottom=np.min(data.geom_xpos[tire_ids,2]-abs(axis)*model.geom_size[tire_ids,1]-model.geom_size[tire_ids,0])
            self.assertAlmostEqual(bottom,-.0005791213481721655,places=6)

    def test_posture_is_sensitive_near_upright(self):
        angle=torch.tensor([0.,.05,.1,.2])
        g=torch.stack((torch.zeros_like(angle),-angle.sin(),-angle.cos()),dim=1)
        reward=posture_reward(g)
        self.assertEqual(float(reward[0]),1.)
        self.assertTrue(bool((reward[:-1]>reward[1:]).all()))
        self.assertLess(float(reward[-1]),.02)

    def test_linkage_cost_is_symmetric_and_leaves_small_corrections_free(self):
        x=torch.tensor([0.,.2,-.2,math.pi/6,-math.pi/6])
        result=limit_cost(x,math.pi/6)
        torch.testing.assert_close(result[:3],torch.zeros(3))
        self.assertGreater(float(result[3]),0.)
        self.assertEqual(float(result[3]),float(result[4]))

    def test_gate_rejects_missing_trials_short_runs_and_failed_cases(self):
        report={'phase':'moving','duration_s':15.,'trials_per_case':16,
                'cases':[{'name':name,'survival_rate':1.,'pass_rate':1.,
                          'max_closure_error_m':.0001} for name,_,_ in gate_cases('moving')]}
        self.assertTrue(promotion_passed(report))
        partial=copy.deepcopy(report);partial['motion_fraction']=.1
        self.assertFalse(promotion_passed(partial))
        self.assertTrue(promotion_passed(partial,motion_fraction=.1))
        for key,value in (('duration_s',1.5),('trials_per_case',1)):
            bad=copy.deepcopy(report);bad[key]=value
            self.assertFalse(promotion_passed(bad))
        bad=copy.deepcopy(report);bad['cases'].pop()
        self.assertFalse(promotion_passed(bad))
        for key,value in (('survival_rate',.5),('pass_rate',.5),
                          ('max_closure_error_m',.002),('pass_rate',float('nan'))):
            bad=copy.deepcopy(report);bad['cases'][1][key]=value
            self.assertFalse(promotion_passed(bad))


if __name__=='__main__':
    unittest.main()
