"""Regression checks for the local recovery planner/executor boundary."""
import unittest
import mujoco
from mujoco import rollout
import numpy as np
from probe_rise import Experiment
from optimize.plan_recovery import capture_cost


class LocalRecoveryTests(unittest.TestCase):
    def test_native_plan_matches_control_tick_replay(self):
        ex=Experiment()
        schedule=[[.2,100,-10],[.2,-100,15],[.2,80,0],[.2,-80,-15],[.2,0,0]]
        _,rows,frames=ex.run([8,1.2,.002,10,1.5,1.5],schedule=schedule,
                            capture_angle=0,duration=1,settle=2,record=True)
        targets=np.repeat(np.array(schedule)[:,1:],20,axis=0)
        np.testing.assert_array_equal(np.asarray(rows)[:,10:12],targets)
        d=ex.reset(settle=2)
        spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
        initial=np.empty(mujoco.mj_stateSize(ex.model,spec))
        mujoco.mj_getState(ex.model,d,initial,spec)
        states,_=rollout.rollout(ex.model,mujoco.MjData(ex.model),initial,
            np.repeat(targets,5,axis=0),initial_warmstart=d.qacc_warmstart)
        # Frame 20 is the state at 0.60 s, immediately before a control change.
        np.testing.assert_allclose(frames[20],states[0,299,1:1+ex.model.nq],atol=1e-9,rtol=0)

    def test_capture_momentum_uses_joint_mapping_not_action_order(self):
        ex=Experiment()
        states=np.zeros((2,1,1+ex.model.nq+ex.model.nv))
        states[:,:,3]=.37
        states[:,:,4]=1 # identity quaternion
        states[0,0,1+ex.model.nq+ex.dofs[0]]=100 # flywheel
        states[1,0,1+ex.model.nq+ex.dofs[1]]=100 # rolling wheel
        cost,_=capture_cost(states,ex.model.nq,ex.dofs[0])
        self.assertAlmostEqual(float(cost[0,0]),.12)
        self.assertAlmostEqual(float(cost[1,0]),0.)


if __name__=='__main__': unittest.main()
