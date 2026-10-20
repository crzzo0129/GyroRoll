"""Search bounded wheel-speed trajectories for the tapered-body recovery probe.

Native MuJoCo rollouts perform all dynamics/contact steps. The optimizer only
selects motor targets, starting from a settled, unpowered side-lying state.
This is trajectory planning, not a trained policy or a success certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from mujoco import rollout
import numpy as np
from probe_rise import Experiment


def capture_cost(states, nq, rotor_dof, rate_weight=.045, momentum_weight=.000012,
                 yaw_weight=.01):
    q = states[..., 4:8]
    tilt = np.arccos(np.clip(1-2*(q[..., 1]**2+q[..., 2]**2), -1, 1))
    vel = states[..., 1+nq:]
    # Penalize velocity and rotor momentum at the actual capture state.
    cost = tilt**2 + rate_weight*np.sum(vel[..., 3:5]**2, axis=-1)
    cost += yaw_weight*vel[..., 5]**2 + momentum_weight*vel[..., rotor_dof]**2
    cost += .02*np.sum(vel[..., :2]**2, axis=-1)
    cost += 8*np.maximum(.32-states[..., 3], 0)**2
    return cost, tilt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generations', type=int, default=60)
    p.add_argument('--population', type=int, default=96)
    p.add_argument('--knots', type=int, default=20)
    p.add_argument('--knot-duration', type=float, default=.12)
    p.add_argument('--timestep', type=float, default=.002)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--rate-weight', type=float, default=.045)
    p.add_argument('--momentum-weight', type=float, default=.000012)
    p.add_argument('--yaw-weight', type=float, default=.01)
    p.add_argument('--flywheel-only', action='store_true')
    p.add_argument('--start', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/wheelbot_cargo/chamfer_recovery/plan.json'))
    args = p.parse_args()
    ex = Experiment(timestep=args.timestep)
    m = ex.model
    data = ex.reset(settle=2.)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    initial = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, data, initial, spec)
    nsub = round(args.knot_duration/args.timestep)
    rng = np.random.default_rng(args.seed)
    mean = np.zeros((args.knots, 2))
    std = np.ones_like(mean)*.8
    best_x, best_cost = mean.copy(), float('inf')
    if args.start:
        prev = json.loads(args.start.read_text())
        old = np.asarray(prev['normalized_targets'])
        mean = old[np.minimum(np.arange(args.knots), len(old)-1)].copy()
        best_x = mean.copy()
        std[:] = .3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = []
    begin = time.monotonic()
    with rollout.Rollout(nthread=args.threads) as pool:
        datas = [mujoco.MjData(m) for _ in range(args.threads)]
        for gen in range(args.generations):
            x = rng.normal(mean, std, (args.population, args.knots, 2))
            # Maintain multiple basins and correlated whole-body rocking pulses.
            nlocal = args.population//3
            x[:nlocal] = best_x + rng.normal(size=x[:nlocal].shape)*rng.choice([.04,.12,.3])
            if gen % 5 == 0:
                count = args.population//6
                coarse = rng.uniform(-1, 1, (count, (args.knots+3)//4, 2))
                x[-count:] = np.repeat(coarse, 4, axis=1)[:, :args.knots]
            x = np.clip(x, -1, 1)
            x[0] = best_x
            if args.flywheel_only:
                x[:,:,1] = 0.
            controls = np.repeat(x*np.array([250.,35.]), nsub, axis=1)
            states, _ = pool.rollout(m, datas, initial, controls,
                                     initial_warmstart=data.qacc_warmstart.copy())
            costs, tilts = capture_cost(states, m.nq, ex.dofs[0], args.rate_weight,
                                       args.momentum_weight, args.yaw_weight)
            # Only consider control boundaries, so the saved schedule is exact.
            costs = costs[:, nsub-1::nsub]
            costs[:, :2] = 1e6
            invalid = np.any(np.diff(states[..., 0], axis=1) <= 0, axis=1)
            costs[invalid] = 1e6
            ends = np.argmin(costs, axis=1)
            scores = costs[np.arange(args.population), ends]
            order = np.argsort(scores)
            winner = order[0]
            if scores[winner] < best_cost:
                best_cost = float(scores[winner])
                best_x = x[winner].copy()
                end = int(ends[winner])+1
                best = dict(normalized_targets=best_x.tolist(),
                            schedule=[[nsub*args.timestep, *v] for v in (best_x[:end]*[250.,35.]).tolist()],
                            params=[8.,1.2,.002,10.,1.5,1.5], capture_angle=0.,
                            settle_s=2., precharge_s=0., cost=best_cost,
                            planned_capture_s=end*nsub*args.timestep,
                            capture_tilt_deg=float(np.degrees(tilts[winner,end*nsub-1])),
                            physics_timestep=args.timestep,
                            cost_weights=dict(rate=args.rate_weight,momentum=args.momentum_weight,
                                              yaw=args.yaw_weight),
                            geometry_sha256=hashlib.sha256(ex.model_path.read_bytes()).hexdigest(),
                            seed=args.seed, generation=gen)
                args.output.write_text(json.dumps(best, indent=2), encoding='utf-8')
                print(json.dumps({k:best[k] for k in ('generation','cost','planned_capture_s','capture_tilt_deg')}),flush=True)
            elites = x[order[:max(6,args.population//8)]]
            mean = .25*mean + .75*elites.mean(axis=0)
            std = np.maximum(.07, .25*std + .75*elites.std(axis=0))
            log.append(dict(generation=gen,best=best_cost,median=float(np.median(scores))))
            if gen % 5 == 0:
                print(f'Generation {gen}/{args.generations}; elapsed {time.monotonic()-begin:.1f}s',flush=True)
    args.output.with_suffix('.search.json').write_text(json.dumps(log,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
