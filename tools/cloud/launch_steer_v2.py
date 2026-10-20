"""Launch two matched steering fine-tuning runs on idle H100s."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/inspire/qb-ilm2/project/leverage-robot/ky26210/GyroRoll/experiments/structure_rl_steer_v2_20261019')
os.chdir(root)
python='/opt/mjlab-venv/bin/python'
baseline='/inspire/qb-ilm2/project/leverage-robot/ky26210/GyroRoll/experiments/structure_rl_20261019_v1/logs/structure_rl/raked_steer/2026-10-19_compare01/tracking/model_7498.pt'
query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
gpu_map=dict(line.strip().split(', ') for line in query.splitlines() if line.strip())
selected=[gpu_map[str(i)] for i in (0,1)]
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
if any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
    raise RuntimeError('Selected GPU has an existing compute process')
assert (root/'checks/preflight/complete.json').exists()
assert (root/'checks/preflight_evaluation.json').exists()
launch=root/'logs/steer_v2/launch'
launch.mkdir(parents=True,exist_ok=False)
record={'source_checkpoint':baseline,'profile':'steer_v2','jobs':[]}
children=[]
def save():
    temp=launch/'jobs.tmp'
    temp.write_text(json.dumps(record,indent=2),encoding='utf-8')
    temp.replace(launch/'jobs.json')
try:
    for gpu,coeff,name in ((selected[0],0.,'balanced'),(selected[1],1.,'balanced_mirror')):
        output=root/'logs/steer_v2'/name
        log=launch/(name+'.log')
        command=[python,'-u','-m','structure_rl.optimize','--checkpoint',baseline,
                 '--output',str(output),'--mirror-coeff',str(coeff),'--num-envs','256',
                 '--iterations-per-stage','2000']
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                   MUJOCO_EGL_DEVICE_ID='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
        with log.open('w',encoding='utf-8') as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        item={'name':name,'pid':child.pid,'gpu_uuid':gpu,'mirror_coefficient':coeff,
              'command':command,'log':str(log),'status_file':str(output/'status.json'),'exit_code':None}
        record['jobs'].append(item);children.append((child,item));save()
        print(json.dumps(item),flush=True)
    for child,item in children:
        item['exit_code']=child.wait();save()
    if any(item['exit_code'] for child,item in children):
        raise SystemExit('A steering optimization failed: inspect logs')
finally:
    save()
