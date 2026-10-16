import _common
import argparse
import copy
from pathlib import Path
from pft_et.utils.io import load_config
from pft_et.training.trainer import train
from evaluate import evaluate
from aggregate_results import aggregate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',default=['configs/main/upfall.yaml','configs/main/urfall.yaml']);p.add_argument('--eval-only',action='store_true');p.add_argument('--output-root',default='outputs/main');p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args()
    for path in a.configs:
        original=load_config(path);runs=[]
        for seed in original['experiment']['seeds']:
            cfg=copy.deepcopy(original);cfg['training']['seed']=seed
            directory=Path(a.output_root)/cfg['data']['dataset']/cfg['experiment']['name']/f'seed{seed}'
            cfg['experiment']['output_dir']=str(directory);checkpoint=directory/'best.pt'
            if not a.eval_only: train(cfg)
            evaluate(cfg,checkpoint,analysis=True);runs.append(directory)
        aggregate(runs,Path(a.output_root)/cfg['data']['dataset']/cfg['experiment']['name']/'summary',bootstrap=a.bootstrap)
