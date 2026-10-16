import _common
import argparse
import ast
import json
import re
from pathlib import Path
import py_compile
import yaml
from pft_et.utils.io import load_config,read_jsonl,save_json
from pft_et.data.splits import validate_splits

def validate_baseline_registry(path):
    registry=yaml.safe_load(Path(path).read_text());protocol=registry['protocol'];baselines=registry['baselines']
    expected={'tfd','tcnte','pose_vit','dgnn_lstm','dynamic_deephit'}
    if set(baselines)!=expected: raise ValueError('Baseline registry does not contain the five comparison methods')
    budget=protocol['tuning_budget']['candidate_configurations_per_dataset']
    if protocol['observation_frames']!=40 or protocol['event_horizon_seconds']!=3.0:
        raise ValueError('Baseline protocol disagrees with the unified experiment settings')
    if len(protocol['tuning_budget']['final_training_seeds'])!=3:
        raise ValueError('Baseline protocol requires three final training runs')
    for name,entry in baselines.items():
        search=entry['search_space'];count=len(search['learning_rate'])*len(search['weight_decay'])
        if count!=budget: raise ValueError(f'{name} search space disagrees with the tuning budget')
    return registry

def audit(config=None):
    root=Path(__file__).resolve().parents[1];files=list((root/'src').rglob('*.py'))+list((root/'scripts').rglob('*.py'))+list((root/'tests').rglob('*.py'))
    failures=[]
    for path in files:
        py_compile.compile(str(path),doraise=True)
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Pass): failures.append(f'{path.relative_to(root)}: empty implementation')
            if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call) and isinstance(node.exc.func,ast.Name) and node.exc.func.id=='NotImplementedError':
                failures.append(f'{path.relative_to(root)}: unimplemented core')
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('pft_et'):
                local=root/'src'/Path(*node.module.split('.'))
                if not local.with_suffix('.py').exists() and not (local/'__init__.py').exists(): failures.append(f'Missing module {node.module}')
    configs=list((root/'configs').rglob('*.yaml'))
    for path in configs:
        validate_baseline_registry(path) if path.name=='baselines.yaml' else load_config(path)
    readme=(root/'README.md').read_text()
    for command in re.findall(r'python (scripts/[a-z_]+\.py)',readme):
        if not (root/command).exists(): failures.append(f'Missing README command {command}')
    if config:
        cfg=load_config(config);rows=read_jsonl(cfg['data']['manifest']);split=json.loads(Path(cfg['data']['splits']).read_text())
        validate_splits(rows,split,cfg['data']['dataset'],not cfg['experiment']['synthetic'])
    if failures: raise RuntimeError('\n'.join(failures))
    return dict(python_files=len(files),configs=len(configs),syntax=True,local_imports=True,readme_commands=True,empty_core_modules=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output');a=p.parse_args();result=audit(a.config)
    if a.output: save_json(a.output,result)
    print(json.dumps(result,indent=2))
