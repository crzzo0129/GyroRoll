"""Run smoke, two PPO stages and matched evaluation on one assigned GPU."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from .model import MORPHOLOGIES


def latest_checkpoint(directory):
    models = list(directory.glob('model_*.pt'))
    if not models:
        raise RuntimeError(f'No checkpoint in {directory}')
    return max(models, key=lambda p: int(p.stem.split('_')[-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--morphology', choices=MORPHOLOGIES, required=True)
    p.add_argument('--output-root', type=Path, default=Path('logs/structure_rl'))
    p.add_argument('--run-name', default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    p.add_argument('--warmup-iterations', type=int, default=2500)
    p.add_argument('--tracking-iterations', type=int, default=5000)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if min(args.warmup_iterations, args.tracking_iterations, args.num_envs) < 1:
        raise ValueError('Iteration and environment counts must be positive')
    root = args.output_root / args.morphology / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    status = {'morphology': args.morphology, 'status': 'starting', 'checkpoint': None}
    def save():
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temp.replace(root / 'status.json')
    def run(mode, *options):
        command = [sys.executable, '-u', '-m', 'structure_rl', mode, '--morphology', args.morphology,
                   '--device', args.device, '--seed', str(args.seed), *map(str, options)]
        print('RUN', ' '.join(command), flush=True)
        with (root / 'commands.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(command) + '\n')
        subprocess.run(command, check=True)
    save()
    try:
        status['status'] = 'smoke'; save()
        run('smoke')
        first = 'warmup' if args.morphology == 'twin_sync' else 'moving'
        checkpoint = None
        for phase, iterations in ((first, args.warmup_iterations), ('tracking', args.tracking_iterations)):
            status['status'] = 'training_' + phase; save()
            directory = root / phase
            resume = ['--checkpoint', checkpoint] if checkpoint else []
            run('train', '--phase', phase, '--iterations', iterations, '--num-envs', args.num_envs,
                '--run-dir', directory, *resume)
            if not (directory / 'complete.json').exists():
                raise RuntimeError(f'Training did not complete: {directory}')
            checkpoint = latest_checkpoint(directory)
            status.update(checkpoint=str(checkpoint), status='evaluating_' + phase); save()
            run('evaluate', '--checkpoint', checkpoint, '--output', root / f'{phase}_evaluation.json')
        status['status'] = 'complete'; save()
    except BaseException as error:
        status.update(status='failed', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
