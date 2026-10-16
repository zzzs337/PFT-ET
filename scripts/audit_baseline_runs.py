"""Audit baseline search and final-run evidence against the declared registry."""
import argparse
import json
from pathlib import Path
import yaml


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(registry, runs_root):
    spec=yaml.safe_load(Path(registry).read_text(encoding='utf-8'));root=Path(runs_root)
    budget=spec['protocol']['tuning_budget'];required=spec['protocol']['required_evidence'];methods={}
    for method in spec['baselines']:
        errors=[];checks=[];search=root/method/'search';final=root/method/'final'
        for name in required['search']:
            if not (search/name).is_file(): errors.append(f'missing search/{name}')
        records=search/'validation_records.jsonl'
        if records.is_file():
            rows=[json.loads(line) for line in records.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(rows)!=budget['candidate_configurations_per_dataset']: errors.append('search candidate count differs from registry')
            else: checks.append(f"{len(rows)} search candidates recorded")
        selected=search/'selected.json'
        selected_record=load_json(selected) if selected.is_file() else None
        if selected_record:
            if selected_record.get('selection_partition')!='val' or selected_record.get('test_used') is not False:
                errors.append('selection record is not validation-only')
            else: checks.append('validation-only selection recorded')
        trials=sorted(final.glob('trial*')) if final.is_dir() else []
        if len(trials)!=len(budget['final_training_seeds']): errors.append('final trial count differs from registry')
        observed=[]
        for trial in trials:
            candidate=trial/'candidate.json'
            if not candidate.is_file(): errors.append(f'{trial.name}: missing candidate.json');continue
            c=load_json(candidate);observed.append(c.get('seed'))
            if selected_record and c.get('adapter_sha256')!=selected_record.get('adapter_sha256'):
                errors.append(f'{trial.name}: adapter digest differs from selection')
            for name in required['final_trial'][1:]:
                if not (trial/name).is_file(): errors.append(f'{trial.name}: missing {name}')
            for name in required['final_evaluation']:
                if not (trial/'evaluation'/name).is_file(): errors.append(f'{trial.name}: missing evaluation/{name}')
            checkpoints=list((trial/'training').glob('*.pt')) if (trial/'training').is_dir() else []
            if not checkpoints: errors.append(f'{trial.name}: no checkpoint file')
        if trials and observed!=budget['final_training_seeds']: errors.append('final seeds differ from registry order')
        if trials and not errors: checks.append(f"{len(trials)} final runs checked")
        methods[method]={'complete':not errors,'errors':errors,'checks':checks}
    return {'complete':all(v['complete'] for v in methods.values()),'registry':str(registry),'runs_root':str(runs_root),'methods':methods}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registry',default='configs/baselines.yaml');parser.add_argument('--runs-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=audit(args.registry,args.runs_root);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));raise SystemExit(not report['complete'])
