"""Local physical-contract tests; GPU MJLab smoke remains a separate gate."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from structure_rl.model import VERSION, SYNC_TOLERANCE_RAD_S, MotorSettings, load_manifest, make_spec, model_xml, source_model
from structure_rl.pipeline import latest_checkpoint


class StructureRLTests(unittest.TestCase):
    def test_servo_force_limits_and_sign(self):
        for kind in ('twin_sync', 'raked_steer'):
            m = mujoco.MjModel.from_xml_string(model_xml(kind, ground=True))
            d = mujoco.MjData(m)
            for sign in (-1, 1):
                mujoco.mj_resetData(m, d)
                d.ctrl[:] = sign * 1000
                mujoco.mj_forward(m, d)
                expected = np.array([2.4, 1.2, 1.2] if kind == 'twin_sync' else [2.4, 2.4])
                np.testing.assert_allclose(d.actuator_force, expected * sign, atol=1e-10)
            self.assertAlmostEqual(m.actuator_ctrlrange[0, 1], 35.)
            if kind == 'twin_sync':
                np.testing.assert_allclose(m.actuator_ctrlrange[1:, 1], 250.)
            else:
                self.assertAlmostEqual(m.actuator_ctrlrange[1, 1], .7)

    def test_equality_survives_scene_attachment(self):
        scene = mujoco.MjSpec()
        child = make_spec('twin_sync')
        scene.attach(child, prefix='robot/', frame=scene.worldbody.add_frame())
        m = scene.compile()
        self.assertEqual(m.neq, 1)
        np.testing.assert_allclose(m.eq_data[0, :5], [0, 1, 0, 0, 0])
        self.assertEqual(m.eq_obj1id[0], m.joint('robot/left').id)
        self.assertEqual(m.eq_obj2id[0], m.joint('robot/right').id)
        self.assertFalse(any(m.geom(i).name.endswith('ground') for i in range(m.ngeom)))

    def test_synchronized_finite_torque_rollout(self):
        for kind in ('twin_sync', 'raked_steer'):
            m = mujoco.MjModel.from_xml_string(model_xml(kind, ground=True))
            d = mujoco.MjData(m)
            maximum_sync = 0.
            for step in range(1000):
                d.ctrl[0] = 2 * np.sin(step * .01)
                d.ctrl[1:] = (30 if kind == 'twin_sync' else .2) * np.sin(step * .02)
                mujoco.mj_step(m, d)
                self.assertTrue(np.isfinite(d.qvel).all())
                self.assertTrue(np.all(abs(d.actuator_force) <= m.actuator_forcerange[:, 1] + 1e-9))
                np.testing.assert_array_equal(d.xfrc_applied, 0.)
                if kind == 'twin_sync':
                    difference = abs(d.qvel[m.joint('left').dofadr[0]] - d.qvel[m.joint('right').dofadr[0]])
                    maximum_sync = max(maximum_sync, difference)
            # Includes deliberately uncontrolled falls and rotor-ground impacts.
            self.assertLess(maximum_sync, SYNC_TOLERANCE_RAD_S)

    def test_geometry_and_morphology_snapshot_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry = source_model('twin_sync').read_bytes()
            (root / 'geometry.xml').write_bytes(geometry)
            manifest = {'task_version': VERSION, 'morphology': 'twin_sync',
                        'observation_dimension': 16, 'action_dimension': 2,
                        'geometry_sha256': hashlib.sha256(geometry).hexdigest()}
            (root / 'task.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(load_manifest(root / 'model_50.pt')['morphology'], 'twin_sync')
            with self.assertRaisesRegex(ValueError, 'morphology mismatch'):
                load_manifest(root / 'model_50.pt', 'raked_steer')
            (root / 'geometry.xml').write_bytes(geometry + b'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_manifest(root / 'model_50.pt')

    def test_invalid_ratio_and_parameters_rejected(self):
        with self.assertRaises(ValueError):
            MotorSettings(rotor_limit=float('nan'))
        with tempfile.TemporaryDirectory() as tmp:
            import xml.etree.ElementTree as ET
            root = ET.parse(source_model('twin_sync'))
            root.find('equality/joint').set('polycoef', '0 -1 0 0 0')
            path = Path(tmp) / 'opposite.xml'
            root.write(path)
            with self.assertRaisesRegex(ValueError, 'fixed \\+1'):
                model_xml('twin_sync', model_path=path)

    def test_checkpoint_selection_is_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('model_9.pt', 'model_100.pt', 'model_50.pt'):
                (root / name).touch()
            self.assertEqual(latest_checkpoint(root).name, 'model_100.pt')

    def test_action_mapping_uses_joint_names_and_shared_target(self):
        # A lightweight Entity double lets us test the real action class on CPU
        # without pretending this verifies the unavailable MJLab GPU integration.
        @dataclass(kw_only=True)
        class Cfg:
            entity_name: str
        class Term:
            def __init__(self, cfg, env):
                self.cfg = cfg
                self._entity = env.scene[cfg.entity_name]
                self.num_envs = env.num_envs
                self.device = env.device
        manager = types.ModuleType('mjlab.managers.action_manager')
        manager.ActionTermCfg, manager.ActionTerm = Cfg, Term
        spec = importlib.util.spec_from_file_location('structure_rl._test_actions', Path(__file__).parents[1] / 'structure_rl/actions.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'mjlab.managers.action_manager': manager, spec.name: module}):
            spec.loader.exec_module(module)
            for kind, indices in (('twin_sync', {'right': 0, 'drive': 1, 'left': 2}), ('raked_steer', {'steer': 0, 'drive': 1})):
                class Robot:
                    def __init__(self):
                        self.velocity = torch.zeros((2, len(indices)))
                        self.position = torch.zeros_like(self.velocity)
                    def find_joints(self, names, preserve_order):
                        return [indices[n] for n in names], list(names)
                    def set_joint_velocity_target(self, value, joint_ids):
                        self.velocity[:, joint_ids] = value
                    def set_joint_position_target(self, value, joint_ids):
                        self.position[:, joint_ids] = value
                robot = Robot()
                env = types.SimpleNamespace(scene={'robot': robot}, num_envs=2, device='cpu')
                term = module.StructureActionCfg(entity_name='robot', morphology=kind).build(env)
                term.process_actions(torch.tensor([[2., -.5], [-.25, -2.]]))
                term.apply_actions()
                torch.testing.assert_close(robot.velocity[:, indices['drive']], torch.tensor([35., -8.75]))
                if kind == 'twin_sync':
                    torch.testing.assert_close(robot.velocity[:, indices['left']], torch.tensor([-125., -250.]))
                    torch.testing.assert_close(robot.velocity[:, indices['left']], robot.velocity[:, indices['right']], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(robot.position[:, indices['steer']], torch.tensor([-.35, -.7]))
                term.reset(torch.tensor([0]))
                torch.testing.assert_close(term.raw_action[0], torch.zeros(2))
                self.assertNotEqual(float(term.raw_action[1, 0]), 0.)

    def test_evaluation_excludes_failed_slots_after_manual_reset(self):
        from structure_rl.evaluation import evaluate
        robot = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2),
            root_link_lin_vel_b=torch.zeros((2, 3)),
            projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2),
            joint_vel=torch.zeros((2, 2))))
        class Env:
            num_envs, device, step_dt = 2, 'cpu', .5
            scene = {'robot': robot}
            def __init__(self):
                self.pending = torch.zeros(2, dtype=torch.bool)
                self.command = torch.zeros((2, 3))
                self.command_manager = types.SimpleNamespace(get_command=lambda _: self.command)
                self.resets = 0
            def reset(self, env_ids=None):
                if env_ids is None:
                    self.step_index = 0
                    self.pending[:] = False
                else:
                    self.pending[env_ids] = False
                    self.resets += len(env_ids)
            def get_observations(self):
                return {'actor': torch.zeros((2, 16))}
            def step(self, action):
                if self.pending.any():
                    raise RuntimeError('Manual reset required before stepping again')
                self.step_index += 1
                done = torch.tensor([self.step_index == 2, False])
                self.pending |= done
                return {}, torch.zeros(2), done, {}
        env = Env()
        runner = types.SimpleNamespace(get_inference_policy=lambda **_: lambda obs: torch.zeros((2, 2)))
        base_task, concept_task = types.ModuleType('miniwheelbot_rl.task'), types.ModuleType('structure_rl.task')
        base_task.heading_rate = lambda env: torch.zeros(2)
        concept_task.joint_ids = lambda env, morphology: [0, 1]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                'miniwheelbot_rl.task': base_task, 'structure_rl.task': concept_task}):
            args = types.SimpleNamespace(checkpoint=Path('model.pt'), morphology='raked_steer',
                                         duration=4., seed=42, output=Path(tmp) / 'evaluation.json')
            report = evaluate(args, env, env, runner)
            self.assertEqual(len(report['cases']), 12)
            self.assertEqual(env.resets, 12)
            for case in report['cases']:
                self.assertEqual(case['survival_rate'], .5)
                self.assertFalse(case['individual'][0]['passed'])
                self.assertEqual(case['individual'][0]['samples'], 0)
                self.assertEqual(case['individual'][0]['survival_s'], 1.)
            self.assertEqual(report['cases'][0]['pass_rate'], .5)
            self.assertTrue(args.output.with_suffix('.html').exists())


if __name__ == '__main__':
    unittest.main()
