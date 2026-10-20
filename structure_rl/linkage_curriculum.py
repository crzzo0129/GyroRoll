"""Train in evaluated blocks; never promote a failed linkage balance policy."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from .linkage_training import promotion_passed,candidate_acceptable
from .pipeline import latest_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--block-iterations',type=int,default=300)
    p.add_argument('--max-blocks',type=int,default=8)
    p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--profile',choices=('linkage_v2','linkage_v3','linkage_v4'),default='linkage_v2')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--learning-rate',type=float,default=1e-6)
    p.add_argument('--graded-moving',action='store_true',help='Evaluate 10%, 25%, 50%, then full moving lessons')
    p.add_argument('--min-learning-rate',type=float,default=1e-9)
    a=p.parse_args()
    if min(a.block_iterations,a.max_blocks,a.num_envs)<1:raise ValueError('Counts must be positive')
    if not all(math.isfinite(v) and v>0 for v in (a.learning_rate,a.min_learning_rate)):
        raise ValueError('Learning rates must be finite and positive')
    if a.checkpoint:
        manifest=json.loads(a.checkpoint.with_name('task.json').read_text())
        if manifest['training_profile']!=a.profile:
            raise ValueError('Checkpoint and requested neural architecture/profile must match')
    a.output.mkdir(parents=True,exist_ok=False)
    status={'status':'starting','checkpoint':None,'history':[]}
    def save():
        tmp=a.output/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        tmp.replace(a.output/'status.json')
    checkpoint=a.checkpoint
    learning_rate=a.learning_rate
    completed_phases=[]
    def evaluate(checkpoint,phase,path,fraction=1.):
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-u','-m','structure_rl.linkage_gate',
            '--checkpoint',str(checkpoint),'--phase',phase,'--motion-fraction',str(fraction),'--output',str(path)],check=True)
        return json.loads(path.read_text())
    try:
        lessons=[('warmup',1.),*[( 'moving',f) for f in ((.1,.25,.5,1.) if a.graded_moving else (1.,))],('tracking',1.)]
        for phase,fraction in lessons:
            lesson=phase if fraction==1 else f'{phase}_{fraction:g}'
            status.update(phase=phase,motion_fraction=fraction)
            baseline=evaluate(checkpoint,phase,a.output/lesson/'baseline.json',fraction) if checkpoint else None
            if baseline and promotion_passed(baseline,motion_fraction=fraction):
                status['history'].append({'phase':phase,'motion_fraction':fraction,'baseline_passed':True,'checkpoint':str(checkpoint)})
                status.update(checkpoint=str(checkpoint));save()
                completed_phases.append((phase,fraction))
                continue
            for block in range(a.max_blocks):
                directory=a.output/lesson/f'block_{block:02d}'
                status.update(status='training',phase=phase,block=block);save()
                command=[sys.executable,'-u','-m','structure_rl','train','--morphology','roll_linkage',
                    '--profile',a.profile,'--phase',phase,'--motion-fraction',str(fraction),'--iterations',str(a.block_iterations),
                    '--learning-rate',str(learning_rate),'--num-envs',str(a.num_envs),'--seed',str(a.seed),'--run-dir',str(directory)]
                if checkpoint:command+=['--checkpoint',str(checkpoint)]
                with (a.output/'commands.jsonl').open('a') as file:file.write(json.dumps(command)+'\n')
                subprocess.run(command,check=True)
                candidate=latest_checkpoint(directory)
                status.update(status='evaluating',candidate=str(candidate));save()
                path=directory/'gate.json'
                report=evaluate(candidate,phase,path,fraction)
                accepted=baseline is None or candidate_acceptable(report,baseline)
                # A new motion lesson must retain all earlier demonstrated skills.
                guard_results={}
                if accepted:
                    for previous_phase,previous_fraction in completed_phases:
                        guard_name=previous_phase if previous_fraction==1 else f'{previous_phase}_{previous_fraction:g}'
                        guard=evaluate(candidate,previous_phase,directory/f'guard_{guard_name}.json',previous_fraction)
                        guard_results[guard_name]=promotion_passed(guard,motion_fraction=previous_fraction)
                    accepted=all(guard_results.values())
                if accepted:
                    checkpoint=candidate
                    baseline=report
                else:
                    learning_rate*=.5
                status.update(checkpoint=str(checkpoint),learning_rate=learning_rate)
                passed=accepted and promotion_passed(report,motion_fraction=fraction)
                status['history'].append({'phase':phase,'motion_fraction':fraction,'block':block,'checkpoint':str(checkpoint),
                    'promote':passed,'accepted':accepted,'guards':guard_results,'candidate':str(candidate),'next_learning_rate':learning_rate,'report':str(path),'cases':[
                        {k:c[k] for k in ('name','survival_rate','pass_rate','mean_survival_s')} for c in report['cases']]})
                save()
                if passed:
                    completed_phases.append((phase,fraction))
                    break
                if learning_rate<a.min_learning_rate:
                    status.update(status='needs_revision',reason='Repeated regression reached learning-rate floor; best checkpoint preserved');save()
                    return
            else:
                status.update(status='needs_revision',reason='Stage budget exhausted without passing independent evaluation')
                save()
                return
        status.update(status='complete');save()
    except BaseException as error:
        status.update(status='failed',error=str(error));save()
        raise


if __name__=='__main__':
    main()
