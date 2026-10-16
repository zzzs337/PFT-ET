import json
import random
from pathlib import Path
import numpy as np
import torch
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.models.model import PFTET
from pft_et.losses.losses import joint_loss
from pft_et.training.prediction import predict,to_device
from pft_et.metrics.evaluation import select_threshold
from pft_et.utils.checkpoint import save_checkpoint,load_checkpoint
from pft_et.utils.io import save_json,file_sha256,data_fingerprints,write_jsonl,array_fingerprints
from pft_et.utils.provenance import execution_provenance
from pft_et.utils.seed import set_seed

def device_for(cfg):
    device=cfg['training']['device']
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def train(cfg,resume=None):
    directory=Path(cfg['experiment']['output_dir'])
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError('Output directory is nonempty; choose another output or use --resume')
    directory.mkdir(parents=True,exist_ok=True);save_json(directory/'config.json',cfg)
    (directory/'split_ids.json').write_text(Path(cfg['data']['splits']).read_text())
    (directory/'data_manifest.jsonl').write_text(Path(cfg['data']['manifest']).read_text())
    provenance=dict(data_fingerprints=data_fingerprints(cfg),array_fingerprints=array_fingerprints(cfg),execution=execution_provenance(cfg))
    resource=cfg['model']['rgb_checkpoint']
    if cfg['model']['rgb_backend']=='uniformerv2' and Path(resource).is_file():
        provenance['rgb_pretrained_sha256']=file_sha256(resource)
    save_json(directory/'provenance.json',provenance)
    set_seed(cfg['training']['seed'],cfg['training']['deterministic']);device=device_for(cfg)
    dataset=PrefixDataset(cfg,'train',include_future=True);val=PrefixDataset(cfg,'val')
    model=PFTET(cfg,initialize=resume is None).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['training']['epochs'],eta_min=cfg['training']['final_learning_rate'])
    start=0;best=None
    if resume:
        ck=load_checkpoint(resume,model,optimizer,scheduler,device);start=ck['epoch']+1;best=ck['best']
        # Epoch-keyed RNG and data shuffle below provide reproducible epoch-level resumes.
        if ck.get('data_fingerprints') != data_fingerprints(cfg): raise ValueError('Resume data or split bytes changed')
        if ck.get('array_fingerprints') is not None and ck['array_fingerprints']!=array_fingerprints(cfg):raise ValueError('Resume array contents changed')
        if ck['config']!=cfg: raise ValueError('Resume requires the same resolved configuration')
    for epoch in range(start,cfg['training']['epochs']):
        set_seed(cfg['training']['seed']+epoch,cfg['training']['deterministic'])
        epoch_cfg=json.loads(json.dumps(cfg));epoch_cfg['training']['seed']+=epoch
        loader=make_loader(dataset,epoch_cfg,True);model.train();sums={};n=0
        for batch in loader:
            batch=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            out=model(batch);loss,parts=joint_loss(model,out,batch)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward();optimizer.step();model.update_target()
            size=len(batch['label']);n+=size
            for key,value in parts.items(): sums[key]=sums.get(key,0)+float(value)*size
        scheduler.step()
        # Reset MC RNG before validation for comparable checkpoint selection.
        set_seed(cfg['training']['seed']+100000,cfg['training']['deterministic'])
        predictions=predict(model,make_loader(val,cfg),device)
        metrics=select_threshold(predictions,val.records,cfg)
        score=[metrics['event_f1'],-(metrics['far'] if metrics['far'] is not None else 1e30),-metrics['event_time_nll']]
        improved=best is None or tuple(score)>tuple(best)
        if improved: best=score
        row=dict(epoch=epoch,learning_rate=optimizer.param_groups[0]['lr'],train={k:v/n for k,v in sums.items()},validation=metrics)
        with (directory/'training.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        save_checkpoint(directory/'last.pt',model,optimizer,scheduler,epoch,cfg,metrics['threshold'],best,metrics['threshold_selection'])
        if improved:
            save_checkpoint(directory/'best.pt',model,optimizer,scheduler,epoch,cfg,metrics['threshold'],best,metrics['threshold_selection'])
            save_json(directory/'validation.json',metrics)
            write_jsonl(directory/'val_predictions.jsonl',predictions)
        print(f"epoch={epoch+1} loss={sums['total']/n:.4f} val_F1={metrics['event_f1']:.4f} threshold={metrics['threshold']:.2f}",flush=True)
    return directory/'best.pt'
