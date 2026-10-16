"""Check whether a configured research run has its required local inputs."""
import _common
import argparse
import json
from pathlib import Path

from pft_et.data.integrity import validate_records
from pft_et.data.splits import validate_splits
from pft_et.utils.io import load_config,read_jsonl


def validate(config):
    cfg=load_config(config);checks=[];errors=[];warnings=[]
    manifest=Path(cfg['data']['manifest']);splits=Path(cfg['data']['splits']);checkpoint=Path(cfg['model']['rgb_checkpoint'])
    for label,path in (('manifest',manifest),('split',splits),('RGB checkpoint',checkpoint)):
        if not path.is_file(): errors.append(f'missing {label}: {path}')
        else: checks.append(f'{label} present: {path}')
    if manifest.is_file():
        try:
            records=read_jsonl(manifest)
            integrity=validate_records(records,manifest.resolve().parent,cfg['data']['image_size'])
            checks.append(f"processed arrays checked for {integrity['streams']} streams")
            falls=[r for r in records if r['fall']]
            impact=sum(r.get('impact_seconds') is not None for r in falls)
            checks.append(f'impact annotations present for {impact}/{len(falls)} fall streams')
            if impact<len(falls): warnings.append('impact-relative evaluation is limited to fall streams with impact_seconds')
            if splits.is_file():
                split=json.loads(splits.read_text(encoding='utf-8'))
                validate_splits(records,split,cfg['data']['dataset'],check_counts=True)
                checks.append('split membership, counts, strata, and leakage checked')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append(f'data validation: {exc}')
    return {'ready':not errors,'dataset':cfg['data']['dataset'],'errors':errors,'warnings':warnings,'checks':checks}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();report=validate(args.config)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not args.report_only: raise SystemExit(not report['ready'])
