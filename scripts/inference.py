import _common
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from pft_et.models.model import PFTET
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.training.prediction import predict
from pft_et.training.trainer import device_for
from pft_et.metrics.alarms import alarm_events
from pft_et.utils.checkpoint import load_checkpoint
from pft_et.utils.io import load_config,write_jsonl
from pft_et.utils.seed import set_seed
from pft_et.utils.compatibility import check_checkpoint_config

def inference(cfg,checkpoint,input_manifest,output,threshold=None):
    # Input is prepared RGB + pose, with annotations unnecessary for prediction.
    # A temporary one-partition manifest wrapper reuses the exact training preprocessing.
    import tempfile
    from pft_et.utils.io import read_jsonl
    cfg=json.loads(json.dumps(cfg));cfg['data']['manifest']=str(Path(input_manifest).resolve())
    records=read_jsonl(input_manifest)
    with tempfile.TemporaryDirectory() as temporary:
        split=Path(temporary)/'splits.json';split.write_text(json.dumps(dict(train=[r['sequence_id'] for r in records],val=[],test=[])))
        cfg['data']['splits']=str(split)
        # Inference permits arbitrary input stream count and reads no future arrays.
        dataset=PrefixDataset.__new__(PrefixDataset)
        dataset.cfg=cfg;dataset.include_future=False;dataset.records=records;dataset.root=Path(input_manifest).resolve().parent
        dataset.edges=np.rint(np.array(cfg['data']['bin_edges'])*cfg['data']['fps']).astype(int)
        dataset.hf=int(round(cfg['data']['future_seconds']*cfg['data']['fps']));dataset.index=[]
        from pft_et.data.dataset import excluded
        for ri,r in enumerate(records):
            for t in range(cfg['data']['observation_frames']-1,r['frames']):
                if not any(excluded(r,u/cfg['data']['fps']) for u in range(t-cfg['data']['observation_frames']+1,t+1)):
                    dataset.index.append((ri,t))
        device=device_for(cfg);set_seed(cfg['training']['seed']+100000,cfg['training']['deterministic'])
        ck=load_checkpoint(checkpoint,device=device);check_checkpoint_config(cfg,ck['config'])
        model=PFTET(cfg,initialize=False).to(device);model.load_state_dict(ck['model'],strict=True)
        rows=predict(model,make_loader(dataset,cfg),device)
        threshold=ck['threshold'] if threshold is None else threshold
        centers=(np.array(cfg['data']['bin_edges'][:-1])+cfg['data']['bin_edges'][1:])/2
        limit=cfg['data']['bin_edges'].index(cfg['alarm']['horizon_seconds'])
        for r in records:
            seq=[p for p in rows if p['sequence_id']==r['sequence_id']]
            if not seq: continue
            probs=np.array([p['probability'] for p in seq]);risk=probs[:,:limit].sum(-1)
            onsets,candidates,confirmed=alarm_events([p['time'] for p in seq],risk,threshold,cfg['alarm']['k'],cfg['alarm']['w'],cfg['data']['fps'])
            for j,p in enumerate(seq):
                prob=probs[j];p.pop('label',None);p.pop('tau',None);p.pop('transition',None)
                p.update(fall_probability=float(risk[j]),remaining_seconds=float((prob[:-1]*centers).sum()/max(prob[:-1].sum(),1e-12)),
                         candidate=bool(candidates[j]),confirmed=bool(confirmed[j]),alarm_onset=p['time'] in onsets,threshold=threshold)
        write_jsonl(output,rows)
        return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True,help='Prepared manifest.jsonl');p.add_argument('--output',required=True);p.add_argument('--threshold',type=float)
    a=p.parse_args();inference(load_config(a.config),a.checkpoint,a.input,a.output,a.threshold)
