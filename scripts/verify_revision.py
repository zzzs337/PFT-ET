"""Run and record CPU execution checks separately from research experiments."""
import argparse
import ast
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def verify(output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a fresh verification directory')
    out.mkdir(parents=True,exist_ok=True);records=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def run(name,args,required=True):
        started=time.perf_counter();p=subprocess.run([sys.executable,*args],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=p.stdout.replace(str(Path.cwd()),'<project>').replace(sys.executable,'python')
        (out/f'{name}.log').write_text(log,encoding='utf-8')
        records.append(dict(name=name,command=['python',*args],exit_code=p.returncode,elapsed_seconds=time.perf_counter()-started))
        if p.returncode and required:raise RuntimeError('Verification failed: '+name+'; see log')
    files=sorted(Path('src').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))+sorted(Path('tests').glob('*.py'))
    for path in files:ast.parse(path.read_text(),filename=str(path))
    for name in ('evaluate','recompute_evaluation','reference_transition','run_interventions','select_cases','run_baseline_search','benchmark_runtime','export_evaluation_records','inference','train','build_manifest','summarize_dataset','validate_research_inputs','audit_baseline_runs'):
        run('help_'+name,['scripts/'+name+'.py','--help'])
    run('static_audit',['scripts/audit_project.py','--output',str(out/'static_audit.json')])
    run('tests',['-m','pytest','-q','tests/'])
    example=out/'example_data'
    run('example_manifest',['scripts/build_manifest.py','--metadata','examples/data/metadata.example.csv','--output',str(example/'manifest.jsonl')])
    run('example_summary',['scripts/summarize_dataset.py','--manifest',str(example/'manifest.jsonl'),'--splits','examples/data/splits.example.json','--dataset','upfall','--output',str(example/'audit')])
    smoke=out/'smoke';config=str(smoke/'config.json');checkpoint=str(smoke/'run/best.pt')
    run('smoke',['scripts/smoke_test.py','--output',str(smoke)])
    run('recompute',['scripts/recompute_evaluation.py','--run',str(smoke/'run'),'--output',str(out/'recompute.json')])
    reference=out/'reference'
    run('reference_export',['scripts/reference_transition.py','export','--config',config,'--checkpoint',checkpoint,'--output',str(reference)])
    for kind,scale in [('deterministic',1.),('single',1.),('mixture',1.),('mixture',.5),('mixture',2.)]:
        name=f'{kind}_scale{scale:g}';folder=out/name
        run(name+'_fit',['scripts/reference_transition.py','fit','--reference',str(reference),'--kind',kind,'--scale',str(scale),'--epochs','1','--output',str(folder)])
        run(name+'_predict',['scripts/reference_transition.py','predict','--reference',str(reference),'--checkpoint',str(folder/'best.pt'),'--output',str(folder/'test.jsonl')])
        run(name+'_metrics',['scripts/evaluate_shared_transition.py','--targets',str(reference/'test.jsonl'),'--predictions',str(folder/'test.jsonl'),'--output',str(folder/'metrics.json')])
    run('cases',['scripts/select_cases.py','--features',str(reference/'test.jsonl'),'--predictions',str(smoke/'run/test_predictions.jsonl'),'--output',str(out/'cases')])
    run('interventions',['scripts/run_interventions.py','--config',config,'--checkpoint',checkpoint,'--output',str(out/'interventions')])
    for mode in ('original','mixture_mean','shuffle'):
        run('recompute_'+mode,['scripts/recompute_evaluation.py','--run',str(out/'interventions'/mode),'--output',str(out/f'{mode}_recomputed.json')])
    for mode in ('prepared','model'):
        run('timing_'+mode,['scripts/benchmark_runtime.py','--config',config,'--checkpoint',checkpoint,'--mode',mode,'--warmup','2','--repeats','5','--output',str(out/('timing_'+mode))])
    import torch
    report=dict(status='PASS',python=platform.python_version(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        parsed_python_files=len(files),commands=records,synthetic=True,
        scope='CPU unit tests and generated-input execution checks; not a replication of reported benchmark performance',
        not_executed=['UP-Fall/UR Fall training or test evaluation','baseline model training and historical search verification','RTX 3090 raw-input RTMPose end-to-end timing'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='commands'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/verification_2_3');a=p.parse_args();verify(a.output)
