"""Warm-start steering v2; evaluate at 2000 and 4000 new PPO iterations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .pipeline import latest_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mirror-coeff',type=float,default=1.)
    parser.add_argument('--iterations-per-stage',type=int,default=2000)
    parser.add_argument('--num-envs',type=int,default=256)
    args=parser.parse_args()
    if args.iterations_per_stage<1 or args.num_envs<1:
        raise ValueError('Positive iteration and environment counts required')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','source_checkpoint':str(args.checkpoint),
           'profile':'steer_v2','mirror_coefficient':args.mirror_coeff,'num_envs':args.num_envs,
           'iterations_per_stage':args.iterations_per_stage,'completed_stages':[]}
    def save():
        path=root/'status.tmp';path.write_text(json.dumps(state,indent=2),encoding='utf-8')
        path.replace(root/'status.json')
    def run(mode,*options):
        command=[sys.executable,'-u','-m','structure_rl',mode,'--morphology','raked_steer',
                 '--phase','tracking','--profile','steer_v2','--mirror-coeff',str(args.mirror_coeff),*map(str,options)]
        with (root/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(command)+'\n')
        print('RUN',' '.join(command),flush=True)
        subprocess.run(command,check=True)
    save()
    try:
        state['status']='smoke';save();run('smoke')
        checkpoint=args.checkpoint
        for stage in (1,2):
            state.update(status=f'training_{stage}');save()
            directory=root/f'stage{stage}'
            run('train','--checkpoint',checkpoint,'--run-dir',directory,'--iterations',args.iterations_per_stage,
                '--num-envs',args.num_envs,*(['--warm-start'] if stage==1 else []))
            if not (directory/'complete.json').exists():raise RuntimeError('Training did not finish')
            checkpoint=latest_checkpoint(directory)
            state.update(status=f'evaluating_{stage}',checkpoint=str(checkpoint));save()
            run('evaluate','--checkpoint',checkpoint,'--extended-evaluation',
                '--output',root/f'stage{stage}_evaluation.json')
            state['completed_stages'].append(stage);save()
        state['status']='complete';save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
