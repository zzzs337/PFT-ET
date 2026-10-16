import _common
import argparse
import copy
import subprocess
import sys
from pathlib import Path
import yaml
from pft_et.utils.io import load_config,read_jsonl,save_json
from pft_et.data.splits import generate_split

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--split-seeds',nargs='+',type=int,default=[3407,3408,3409])
    p.add_argument('--output-root',default='outputs/split_sensitivity')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--eval-only',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);records=read_jsonl(cfg['data']['manifest'])
    root=Path(a.output_root)/cfg['data']['dataset'];root.mkdir(parents=True,exist_ok=True)
    paths=[]
    for split_seed in a.split_seeds:
        split=root/f'split{split_seed}.json'
        save_json(split,generate_split(records,cfg['data']['dataset'],split_seed))
        for name,strategy in [('pft_et','full'),('current_only','current'),('deterministic_future','deterministic')]:
            condition=copy.deepcopy(cfg)
            condition['data']['splits']=str(split)
            condition['model']['strategy']=strategy
            condition['experiment']['name']=f'split{split_seed}_{name}'
            if strategy=='current': condition['loss'].update(transition=0.,auxiliary=0.)
            path=root/f'split{split_seed}_{name}.yaml';path.write_text(yaml.safe_dump(condition,sort_keys=False));paths.append(str(path))
    if not a.prepare_only:
        command=[sys.executable,'scripts/run_experiments.py','--configs',*paths,'--output-root',str(root/'runs')]
        if a.eval_only: command.append('--eval-only')
        subprocess.run(command,check=True)
    print('Prepared alternative split configs: '+', '.join(paths))
