"""Reproducible local search of bounded two-wheel recovery pulses."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from probe_rise import Experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=Path)
    p.add_argument("--torque-limit", type=float, default=2.4)
    p.add_argument("--flywheel-limit", type=float, default=250.)
    p.add_argument("--output", type=Path, default=Path("results/local_rise/search.json"))
    args = p.parse_args()
    ex = Experiment(args.torque_limit, args.flywheel_limit)
    rng = np.random.default_rng(args.seed)
    limit=args.flywheel_limit
    bounds = np.array([[.18,.5],[.10,.65],[-35,35],[-35,35],[.05,.5],[-limit,limit],[-35,35],
                       [4,25],[.5,5],[-.005,.008],[4,25],[.3,4],[.1,3],[5,45],[-.8,.8],[-2,2]])
    best, best_x, trials = 1e9, None, []
    if args.start:
        report = json.loads(args.start.read_text())
        s=report['schedule']
        best_x=np.array([s[0][0],s[1][0],s[0][2],s[1][2],s[2][0],s[2][1],s[2][2],
                         *report['params'], report['capture_angle'],report.get('yaw_gain',0.),report.get('gyro_gain',0.)])
    for i in range(args.trials):
        if i==0 and best_x is not None:
            x=best_x.copy()
        elif best_x is not None and i % 5:
            x = np.clip(best_x + rng.normal(size=len(bounds)) * (bounds[:,1]-bounds[:,0]) * rng.choice([.03,.08,.15]),
                        bounds[:,0], bounds[:,1])
        else:
            x = rng.uniform(bounds[:,0], bounds[:,1])
        schedule = [[x[0],-limit,x[2]],[x[1],limit,x[3]],[x[4],x[5],x[6]]]
        r,_,_ = ex.run(x[7:13],schedule=schedule,duration=5,capture_angle=x[13],yaw_gain=x[14],gyro_gain=x[15])
        cost = r["best_capture_cost"] + .05*r["objective"] - 2*r['max_stable_seconds']
        if r["within_5s"]:
            cost -= 100
        if r["stable_at_end"]:
            cost -= 100
        trials.append({"trial":i,"cost":cost,"report":r})
        if cost < best:
            best, best_x = cost, x.copy()
            print(json.dumps({"trial":i,"cost":cost,"min_tilt":r['minimum_tilt_deg'],
                              "recovered":r['within_5s'],"stable":r['stable_at_end']}),flush=True)
        if i % 20 == 0 or i == args.trials-1:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({"seed":args.seed,"trials":trials},indent=2))
            print(f"Completed {i+1}/{args.trials}",flush=True)


if __name__ == '__main__':
    main()
