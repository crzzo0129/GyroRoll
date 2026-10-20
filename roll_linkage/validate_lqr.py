"""Matched finite-actuator LQR trials; retain successful and failed trajectories."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import mujoco

from .lqr import Plant, FullController, trial, OUT, ROOT, DT

FOLDER = OUT / "level_torso_v2"


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    nominal = Plant()
    xml = nominal.xml.encode("utf-8")
    (FOLDER/"model.xml").write_bytes(xml)
    (ROOT/"roll_linkage/model_level_torso.xml").write_bytes(xml)
    controllers = {speed: FullController(nominal, speed, effort=100., yaw_weight=0., phases=8)
                   for speed in (0., .5, 1., -.5, -1.)}
    controller_data = {
        str(speed): dict(A=c.a.tolist(), B=c.b.tolist(), K=c.k.tolist(),
                         equilibrium_ctrl=c.utrim.tolist(), drift=c.drift.tolist(),
                         closed_loop_spectral_radius=c.radius, tangent_indices=c.indices.tolist())
        for speed, c in controllers.items()
    }
    (FOLDER/"controllers.json").write_text(json.dumps(controller_data, indent=2), encoding="utf-8")
    d = nominal.lift(np.zeros(10))
    original = Plant(original=True)
    d0 = original.lift(np.zeros(10))
    checks = dict(mass_kg=float(nominal.m.body_mass.sum()),
                  original_mass_kg=float(original.m.body_mass.sum()),
                  original_com_offset_xy_mm=(d0.subtree_com[original.base, :2]*1000).tolist(),
                  new_com_offset_xy_mm=(d.subtree_com[nominal.base, :2]*1000).tolist(),
                  actuators=int(nominal.m.nu), free_root=True, world_orientation_constraint=False)
    # Explicitly verify independent upper/lower orientation on the joint manifold.
    x = np.zeros(10); x[7] = math.radians(20.)
    tilted = nominal.lift(x)
    checks["kinematic_torso_roll_deg"] = math.degrees(nominal.state(tilted)[0])
    r = tilted.xmat[nominal.carrier].reshape(3, 3)
    checks["kinematic_wheel_carrier_roll_deg"] = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    assert abs(checks["kinematic_torso_roll_deg"]) < 1e-8
    assert abs(checks["kinematic_wheel_carrier_roll_deg"]-20) < 1e-8
    assert abs(checks["mass_kg"]-checks["original_mass_kg"]) < 1e-12
    alternate = FullController(nominal, 0., effort=100., yaw_weight=0., phases=8, epsilon=1e-4)
    checks["finite_difference_A_relative_change"] = float(
        np.linalg.norm(alternate.a-controllers[0.].a)/np.linalg.norm(controllers[0.].a))
    checks["finite_difference_B_relative_change"] = float(
        np.linalg.norm(alternate.b-controllers[0.].b)/np.linalg.norm(controllers[0.].b))
    cases = []
    for speed in controllers:
        for tilt in (-1., -.5, -.1, .1, .5, 1.):
            cases.append(dict(name=f"moving_{speed:+.1f}_roll_{tilt:+.1f}" if speed else f"stand_roll_{tilt:+.1f}",
                              speed=speed, initial_speed=speed, initial_roll=tilt, push=0.))
    cases.append(dict(name="stand_exact", speed=0., initial_speed=0., initial_roll=0., push=0.))
    for force in (-1., -.2, -.05, .05, .2, 1.):
        cases.append(dict(name=f"stand_push_{force:+.2f}", speed=0., initial_speed=0., initial_roll=0., push=force))
    for speed in (.5, -.5):
        cases.append(dict(name=f"start_{speed:+.1f}", speed=speed, initial_speed=0., initial_roll=.1, push=0.))
    results = []
    for timestep in (.001, .0005):
        plant = Plant(timestep=timestep)
        for case in cases:
            c = controllers[case["speed"]]
            result, telemetry, poses = trial(plant, c, duration=15., **case)
            result["controller_linearized_timestep"] = .001
            stem = f'{case["name"]}_dt{timestep:g}'
            result["trajectory"] = stem+".npz"
            frame_rows = telemetry[::4]
            if len(poses) > len(frame_rows):
                frame_rows = np.vstack((frame_rows, telemetry[-1]))
            assert len(frame_rows) == len(poses)
            np.savez_compressed(FOLDER/(stem+".npz"), qpos=poses, telemetry=telemetry, frame_telemetry=frame_rows)
            assert result["peak_torque_Nm"] <= 2.400001
            results.append(result)
            print(stem, "survived", result["survived"], "level", result["level_pass"],
                  "seconds", round(result["duration_s"], 2), flush=True)
    report = dict(model="roll_linkage_level_torso_v2", model_sha256=hashlib.sha256(xml).hexdigest(),
                  checks=checks, duration_s=15, policy_dt_s=DT, cases=results,
                  method="Full-state, phase-averaged local discrete LQR; native velocity/position servos",
                  discount=.999, effort_multiplier=100., wheel_phase_samples=8, yaw_weight=0.,
                  independent_local_controllers=list(controllers),
                  transition_speed_slew_m_s2=.3,
                  limits=dict(drive_command_rad_s=35, roll_command_deg=30, motor_torque_Nm=2.4),
                  pass_rule=dict(survive_s=15, torso_rms_roll_pitch_deg=3,
                                 torso_peak_roll_pitch_deg=5, speed_rmse_m_s=.15,
                                 relative_wheel_peak_deg=31, loop_error_mm=1),
                  recording="torso metrics at 100 Hz; motor torque and relative roll peaks each physics step",
                  assumptions=["flat ground; no payload; exact full simulation state",
                               "no state resets, fixed root, orientation lock or hidden support during trials",
                               "lateral external force only in explicitly named push trials: 5 to 5.2 seconds",
                               "joint limits are compliant; command limits are clipped",
                               "support and linkage self-collisions disabled as in concept V1",
                               "no independent yaw motor; straight heading is measured, not guaranteed",
                               "moving_* starts with assigned initial velocity; start_* begins at rest"],
                  conclusion="Local recovery exists, but level-torso robustness is not established by these tests.")
    (FOLDER/"evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(tests=len(results), survived=sum(r["survived"] for r in results),
                          level_pass=sum(r["level_pass"] for r in results), checks=checks), indent=2))


if __name__ == "__main__":
    main()
