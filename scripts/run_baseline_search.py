"""Validation-only search around a supplied baseline training/prediction adapter."""
import _common
import argparse
import copy
import importlib.util
import itertools
import json
import time
from pathlib import Path
import yaml
from pft_et.utils.io import load_config,save_json,file_sha256,read_jsonl,write_jsonl
from pft_et.utils.provenance import execution_provenance
from evaluate import evaluate


def candidate_grid(space):
    keys=sorted(space)
    return [dict(zip(keys,values)) for values in itertools.product(*(space[k] for k in keys))]


def execute(cfg,method,adapter,output,final_from=None):
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose a fresh baseline output')
    root.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('baseline_adapter',adapter);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not callable(getattr(module,'fit',None)) or not callable(getattr(module,'predict',None)):
        raise ValueError('Adapter must provide fit(config, params, seed, output) -> checkpoint and predict(config, checkpoint, partition, output) -> predictions JSONL')
    protocol=yaml.safe_load(Path('configs/baselines.yaml').read_text());method_cfg=protocol['baselines'][method];budget=protocol['protocol']['tuning_budget']
    candidates=candidate_grid(method_cfg['search_space'])
    if len(candidates)!=budget['candidate_configurations_per_dataset']:raise ValueError('Candidate count disagrees with declared budget')
    records=[];base=copy.deepcopy(cfg);base['experiment']['name']=method
    base['training'].update(epochs=budget['max_epochs'],batch_size=budget['batch_size'])
    if final_from:
        selected=json.loads(Path(final_from).read_text())
        if selected['method']!=method or selected['dataset']!=cfg['data']['dataset']:raise ValueError('Wrong selected baseline settings')
        if selected['adapter_sha256']!=file_sha256(adapter):raise ValueError('Adapter changed after search')
        jobs=[(selected['params'],s) for s in budget['final_training_seeds']]
    else:jobs=[(params,cfg['training']['seed']) for params in candidates]
    for i,(params,seed) in enumerate(jobs):
        trial=root/f'trial{i:03d}';trial.mkdir();c=copy.deepcopy(base);c['training']['seed']=seed
        c['experiment']['output_dir']=str(trial/'training')
        # Search adapters receive only train/val records and cannot read the test membership through these inputs.
        if not final_from:
            splits=json.loads(Path(c['data']['splits']).read_text());allowed=set(splits['train']+splits['val'])
            masked=trial/'search_inputs';masked.mkdir()
            original_root=Path(c['data']['manifest']).resolve().parent
            rows=[]
            for r in read_jsonl(c['data']['manifest']):
                if r['sequence_id'] in allowed:
                    r=dict(r)
                    for k in ('rgb_path','skeleton_path'):r[k]=str((original_root/r[k]).resolve())
                    rows.append(r)
            write_jsonl(masked/'manifest.jsonl',rows);save_json(masked/'splits.json',dict(train=splits['train'],val=splits['val'],test=[]))
            adapter_cfg=copy.deepcopy(c);adapter_cfg['data'].update(manifest=str(masked/'manifest.jsonl'),splits=str(masked/'splits.json'))
        else:adapter_cfg=c
        save_json(trial/'candidate.json',dict(params=params,seed=seed,training_budget=budget,adapter_sha256=file_sha256(adapter),config=c))
        started=time.perf_counter();checkpoint=Path(module.fit(adapter_cfg,params,seed,str(trial/'training')))
        if not checkpoint.is_file():raise FileNotFoundError('Adapter did not produce a checkpoint')
        val_path=module.predict(adapter_cfg,str(checkpoint),'val',str(trial/'val_predictions.jsonl'))
        val=evaluate(c,None,'val',str(val_path),output=trial/'evaluation')
        record=dict(candidate=i,method=method,dataset=cfg['data']['dataset'],params=params,seed=seed,
             validation_event_f1=val['event_f1'],validation_far=val['far'],threshold=val['threshold'],
             elapsed_training_and_validation_seconds=time.perf_counter()-started,checkpoint_sha256=file_sha256(checkpoint))
        if final_from:
            test_path=module.predict(adapter_cfg,str(checkpoint),'test',str(trial/'test_predictions.jsonl'))
            evaluate(c,None,'test',str(test_path),output=trial/'evaluation',selected_threshold=val['threshold'])
        records.append(record);write_jsonl(root/'validation_records.jsonl',records)
    if not final_from:
        selected=max(records,key=lambda r:(r['validation_event_f1'],-(r['validation_far'] if r['validation_far'] is not None else float('inf')),-r['candidate']))
        save_json(root/'selected.json',dict(selected,adapter_sha256=file_sha256(adapter),selection_partition='val',test_used=False))
    save_json(root/'search_context.json',dict(method=method,declared_budget=budget,actual_configurations=len(jobs),final_runs=bool(final_from),
        adapter_sha256=file_sha256(adapter),execution=execution_provenance(cfg)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',required=True)
    p.add_argument('--adapter',required=True);p.add_argument('--output',required=True);p.add_argument('--final-from');a=p.parse_args()
    execute(load_config(a.config),a.method,a.adapter,a.output,a.final_from)
