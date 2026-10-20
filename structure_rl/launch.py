"""Launch both pipelines on two idle physical GPUs; keep a durable job record."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1', help='Physical nvidia-smi GPU indices: twin,steer')
    p.add_argument('--output-root', type=Path, default=Path('logs/structure_rl'))
    p.add_argument('--run-name', default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    p.add_argument('--warmup-iterations', type=int, default=2500)
    p.add_argument('--tracking-iterations', type=int, default=5000)
    p.add_argument('--num-envs', type=int, default=256)
    args = p.parse_args()
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(x.isdigit() for x in gpu_ids):
        raise ValueError('--gpus must contain two different physical GPU indices')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    gpu_map = dict(line.strip().split(', ') for line in query.splitlines() if line.strip())
    selected = [gpu_map[x] for x in gpu_ids]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
        raise RuntimeError('A selected GPU already has a compute process; inspect usage before launch')
    root = args.output_root / ('launch_' + args.run_name)
    root.mkdir(parents=True, exist_ok=False)
    record = {'run_name': args.run_name, 'jobs': []}
    jobs = []
    def save():
        (root / 'jobs.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    try:
        for kind, gpu in zip(('twin_sync', 'raked_steer'), selected):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                       MUJOCO_EGL_DEVICE_ID='0', PYTHONUNBUFFERED='1')
            command = [sys.executable, '-u', '-m', 'structure_rl.pipeline', '--morphology', kind,
                       '--output-root', str(args.output_root), '--run-name', args.run_name,
                       '--warmup-iterations', str(args.warmup_iterations),
                       '--tracking-iterations', str(args.tracking_iterations), '--num-envs', str(args.num_envs)]
            log = root / (kind + '.log')
            with log.open('w', encoding='utf-8') as output:
                child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
            item = {'morphology': kind, 'gpu_uuid': gpu, 'pid': child.pid, 'log': str(log),
                    'status_file': str(args.output_root / kind / args.run_name / 'status.json'), 'exit_code': None}
            record['jobs'].append(item)
            jobs.append((child, item))
            save()
            print(json.dumps(item), flush=True)
        failed = False
        for child, item in jobs:
            item['exit_code'] = child.wait()
            failed |= item['exit_code'] != 0
            save()
        if failed:
            raise SystemExit('At least one training pipeline failed; inspect jobs.json and its log')
    finally:
        save()


if __name__ == '__main__':
    main()
