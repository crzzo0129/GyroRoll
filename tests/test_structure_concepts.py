"""Physical contracts for the independent morphology comparison."""
import unittest
import mujoco
import numpy as np

from probe_structures import Plant, DT


class StructureConceptTests(unittest.TestCase):
    def test_twin_rotors_have_parallel_axes_and_fixed_total_budget(self):
        p=Plant('twin')
        self.assertEqual(p.m.nu,3)
        self.assertAlmostEqual(float(p.m.body_mass.sum()),1.465,places=6)
        axes=np.array([p.m.joint(n).axis for n in ('left','right')])
        np.testing.assert_allclose(axes,[[1,0,0],[1,0,0]])
        self.assertEqual(np.linalg.matrix_rank(axes),1)
        self.assertAlmostEqual(sum(p.m.actuator_ctrlrange[1:,1]),2.4)
        # Rotors must clear the ground and the tire in the upright configuration.
        d=p.reset()
        for name in ('left','right'):
            gid=p.m.geom('flywheel_'+name).id
            center=d.geom_xpos[gid]
            self.assertGreater(center[2]-.055,.06)
            self.assertGreater(abs(center[1])-.055,.012)

    def test_rake_and_ground_trail_match_specification(self):
        p=Plant('steer',30.,.03)
        self.assertEqual(p.m.nu,2)
        self.assertEqual(p.m.njnt,3) # free root, steering, rolling
        d=p.reset()
        axis=d.xaxis[p.m.joint('steer').id]
        pivot=d.xanchor[p.m.joint('steer').id]
        intersection=pivot[0]-pivot[2]*axis[0]/axis[2]
        self.assertAlmostEqual(np.degrees(np.arccos(axis[2])),30.)
        self.assertAlmostEqual(intersection,.03,places=3)

    def test_finite_actuation_and_no_hidden_forces(self):
        for kind in ('twin','steer'):
            p=Plant(kind)
            d=p.reset()
            p.input(d,np.array([100.,-100.]))
            mujoco.mj_step(p.m,d,nstep=round(DT/p.m.opt.timestep))
            self.assertTrue(np.isfinite(d.qpos).all())
            self.assertTrue(np.all(abs(d.actuator_force)<=p.m.actuator_ctrlrange[:,1]+1e-12))
            np.testing.assert_array_equal(d.xfrc_applied,0.)
            np.testing.assert_array_equal(d.qfrc_applied,0.)

    def test_opposite_rotor_torques_do_not_produce_direct_yaw_authority(self):
        p=Plant('twin')
        d=p.reset()
        d.qpos[2]=2.0
        p.m.opt.gravity[:]=0
        d.ctrl[:]=[0.,1.,-1.]
        mujoco.mj_forward(p.m,d)
        self.assertEqual(d.ncon,0)
        np.testing.assert_allclose(d.qacc[3:6],0.,atol=1e-9)
        d.ctrl[:]=[0.,1.,1.]
        mujoco.mj_forward(p.m,d)
        self.assertLess(d.qacc[3],-1.)
        self.assertLess(abs(d.qacc[5]),1e-8)

    def test_explicit_joint_ratios_and_torque_mapping(self):
        for sign in (1,-1):
            p=Plant('twin',coupling=sign)
            self.assertEqual(p.m.neq,1)
            np.testing.assert_allclose(p.m.eq_data[0,:5],[0,sign,0,0,0])
            d=p.reset()
            p.input(d,np.array([.4,2.]))
            np.testing.assert_allclose(d.ctrl,[.4,1.,sign])
            # A free-space spin-up isolates the gear relation from impacts.
            d.qpos[2]=2.;p.m.opt.gravity[:]=0
            residual=0.
            for _ in range(100):
                mujoco.mj_step(p.m,d)
                residual=max(residual,abs(d.qvel[p.dofs[1]]-sign*d.qvel[p.dofs[2]]))
            self.assertLess(residual,1e-6)


if __name__=='__main__': unittest.main()
