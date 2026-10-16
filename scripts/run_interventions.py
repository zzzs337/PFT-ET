"""Same-checkpoint inference interventions with explicit cross-sequence donor maps."""
import _common
import argparse
from pathlib import Path
import numpy as np
import torch
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.models.model import PFTET
from pft_et.training.trainer import device_for
from pft_et.training.prediction import to_device
from pft_et.utils.io import load_config,write_jsonl,save_json,data_fingerprints,file_sha256
from pft_et.utils.checkpoint import load_checkpoint
from pft_et.utils.seed import set_seed
from pft_et.utils.compatibility import check_checkpoint_config
from evaluate import evaluate


def donor_indices(physical_ids,seed):
    groups=sorted(set(physical_ids))
    if len(groups)<2:raise ValueError('Shuffling requires two different physical sequences')
    rng=np.random.default_rng(seed);groups=list(rng.permutation(groups))
    pools={g:np.where(np.array(physical_ids)==g)[0] for g in groups}
    mapping={g:groups[(j+1)%len(groups)] for j,g in enumerate(groups)}
    return [int(rng.choice(pools[mapping[g]])) for g in physical_ids]


@torch.no_grad()
def run(cfg,checkpoint,output):
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose new intervention output')
    root.mkdir(parents=True,exist_ok=True);device=device_for(cfg);ck=load_checkpoint(checkpoint,device=device)
    check_checkpoint_config(cfg,ck['config'])
    if ck['data_fingerprints']!=data_fingerprints(cfg):raise ValueError('Changed data/split')
    if cfg['model']['strategy']!='full':raise ValueError('Interventions require the full model')
    model=PFTET(cfg,initialize=False).to(device);model.load_state_dict(ck['model']);model.eval()
    for partition in ('val','test'):
        ds=PrefixDataset(cfg,partition);metadata={r['sequence_id']:r for r in ds.records};states=[];base=[]
        for batch in make_loader(ds,cfg):
            b=to_device(batch,device);z=model.online(b['rgb'],b['skeleton']);states.append(z.cpu())
            base.extend(dict(sequence_id=b['sequence_id'][i],frame=int(b['frame'][i]),time=float(b['time'][i]),label=int(b['label'][i]),tau=float(b['tau'][i])) for i in range(len(z)))
        states=torch.cat(states);ids=[metadata[r['sequence_id']]['physical_id'] for r in base]
        donors=donor_indices(ids,cfg['training']['seed']+700000)
        write_jsonl(root/f'{partition}_donor_map.jsonl',[dict(sequence_id=r['sequence_id'],frame=r['frame'],
            donor_sequence_id=base[j]['sequence_id'],donor_frame=base[j]['frame'],donor_physical_id=ids[j]) for r,j in zip(base,donors)])
        for mode in ('original','mixture_mean','shuffle'):
            rows=[];set_seed(cfg['training']['seed']+100000,cfg['training']['deterministic'])
            for start in range(0,len(base),cfg['training']['batch_size']):
                end=min(start+cfg['training']['batch_size'],len(base));z=states[start:end].to(device)
                dist=model.transition(states[donors[start:end]].to(device)) if mode=='shuffle' else None
                out=model.from_state(z,distribution=dist,intervention='mixture_mean' if mode=='mixture_mean' else None)
                rows.extend(dict(r,probability=p.cpu().tolist()) for r,p in zip(base[start:end],out['probability']))
            folder=root/mode;folder.mkdir(exist_ok=True);raw=folder/f'{partition}_raw.jsonl';write_jsonl(raw,rows)
            threshold=None
            if partition=='test':
                import json
                threshold=json.loads((folder/'val_metrics.json').read_text())['threshold']
            evaluate(cfg,None,partition,str(raw),output=folder,selected_threshold=threshold)
    save_json(root/'interventions.json',dict(checkpoint_sha256=file_sha256(checkpoint),seed=cfg['training']['seed'],
       model_capacity='unchanged',mixture_mean='one deterministic weighted mixture mean; differs from per-component mean-only',
       shuffle='donor predicted distribution from another physical sequence in the same partition; current state unchanged',
       threshold_policy='validation selection separately for each intervention, then frozen on test',future_observations_at_inference=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(load_config(a.config),a.checkpoint,a.output)
