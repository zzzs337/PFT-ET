import _common
import argparse
import json
from pathlib import Path
from pft_et.utils.io import load_config,save_json
from evaluate import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full-run',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=load_config(a.config)
    full=json.loads((Path(a.full_run)/'validation.json').read_text())
    fullcfg=json.loads((Path(a.full_run)/'config.json').read_text())
    if fullcfg['training']['seed']!=cfg['training']['seed'] or fullcfg['data']['splits']!=cfg['data']['splits']:
        raise ValueError('FAR matched comparison requires corresponding seed and split')
    val=evaluate(cfg,a.checkpoint,'val',far_budget=full['far'],output=a.output)
    save_json(Path(a.output)/'budget.json',dict(reference_validation_far=full['far'],selected_threshold=val['threshold']))
    evaluate(cfg,a.checkpoint,'test',output=a.output,selected_threshold=val['threshold'],analysis=True)
