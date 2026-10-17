"""Replay the same recovery controller across sides and integration steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from probe_rise import Experiment
from wheelbot.recovery_control import PredictiveRecovery


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',type=Path,default=Path('results/chamfer_recovery/tuned.json'))
    p.add_argument('--reference',type=Path,default=Path('results/chamfer_recovery/reference.npz'))
    p.add_argument('--timesteps',type=float,nargs='+',default=[.002,.001,.0005])
    p.add_argument('--rolls',type=float,nargs='+',default=[90,-90])
    p.add_argument('--pitch',type=float,default=0)
    p.add_argument('--duration',type=float,default=15)
    p.add_argument('--output-dir',type=Path,default=Path('results/chamfer_recovery/validation'))
    args=p.parse_args()
    config=json.loads(args.controller.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for dt in args.timesteps:
        for roll in args.rolls:
            ex=Experiment(timestep=dt)
            ctl=PredictiveRecovery(args.reference,ex.model,ex.model_path,side=roll)
            begin=time.perf_counter()
            try:
                report,rows,frames=ex.run(config['params'],roll=roll,pitch=args.pitch,
                    duration=args.duration,record=True,settle=2.,
                    gyro_gain=config.get('gyro_gain',0),roll_gyro_gain=config.get('roll_gyro_gain',0),
                    trajectory_controller=ctl)
            finally:
                ctl.close()
            report.update(trajectory_feedback='mpc',reference=str(args.reference),
                          wall_seconds=time.perf_counter()-begin,replanning=ctl.records,
                          geometry_sha256=hashlib.sha256(ex.model_path.read_bytes()).hexdigest())
            stem=f'dt_{dt:g}_roll_{roll:g}_pitch_{args.pitch:g}'
            (args.output_dir/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            np.savez_compressed(args.output_dir/f'{stem}.npz',trace=rows,qpos=frames)
            result={k:report[k] for k in ('physics_timestep','initial_roll_deg','initial_pitch_deg',
                    'minimum_tilt_deg','recovered_at_s','max_stable_seconds','stable_at_end',
                    'final_tilt_deg','wall_seconds')}
            summaries.append(result)
            (args.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
