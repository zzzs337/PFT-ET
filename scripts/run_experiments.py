import _common
import argparse
import copy
import json
from pathlib import Path
from pft_et.utils.io import load_config
from pft_et.training.trainer import train
from evaluate import evaluate
from aggregate_results import aggregate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',required=True);p.add_argument('--partition',choices=['val','test'],default='test');p.add_argument('--eval-only',action='store_true');p.add_argument('--checkpoint-root',help='For N_MC sensitivity, point to the full-model directory containing seed*/best.pt');p.add_argument('--output-root',default='outputs/experiments')
    a=p.parse_args()
    for path in a.configs:
        original=load_config(path);runs=[]
        for seed in original['experiment']['seeds']:
            cfg=copy.deepcopy(original);cfg['training']['seed']=seed
            directory=Path(a.output_root)/cfg['experiment']['name']/f'seed{seed}';cfg['experiment']['output_dir']=str(directory)
            checkpoint=Path(a.checkpoint_root)/f'seed{seed}'/'best.pt' if a.checkpoint_root else directory/'best.pt'
            if not a.eval_only: train(cfg)
            evaluate(cfg,checkpoint,a.partition,output=directory,analysis=True);runs.append(directory)
        aggregate(runs,Path(a.output_root)/cfg['experiment']['name']/'summary',a.partition)
