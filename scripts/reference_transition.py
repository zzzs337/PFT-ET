"""Export frozen reference coordinates and fit controlled transition predictors."""
import _common
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.models.model import PFTET
from pft_et.models.transition import ConditionalFutureTransition,mixture_log_prob
from pft_et.training.prediction import to_device
from pft_et.training.trainer import device_for
from pft_et.utils.io import load_config,save_json,read_jsonl,write_jsonl,data_fingerprints,file_sha256,array_fingerprints
from pft_et.utils.provenance import execution_provenance
from pft_et.utils.checkpoint import load_checkpoint
from pft_et.utils.seed import set_seed


def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()): raise FileExistsError('Choose a new output directory')
    path.mkdir(parents=True,exist_ok=True);return path


@torch.no_grad()
def export(cfg,checkpoint,output,encoder='target'):
    out=fresh(output);device=device_for(cfg);ck=load_checkpoint(checkpoint,device=device)
    if ck['data_fingerprints']!=data_fingerprints(cfg): raise ValueError('Reference checkpoint uses different data/split')
    if ck.get('array_fingerprints') is not None and ck['array_fingerprints']!=array_fingerprints(cfg):raise ValueError('Reference arrays differ from training')
    for key in ('fps','observation_frames','future_seconds','sampled_frames','image_size','rgb_mean','rgb_std'):
        if cfg['data'][key]!=ck['config']['data'][key]:raise ValueError('Reference preprocessing differs from checkpoint: '+key)
    model=PFTET(cfg,initialize=False).to(device);model.load_state_dict(ck['model']);model.eval()
    ref=getattr(model,encoder);ref.eval();ref.requires_grad_(False);all_rows={}
    for part in ('train','val','test'):
        ds=PrefixDataset(cfg,part,include_future=True);rows=[]
        for b in make_loader(ds,cfg):
            b=to_device(b,device);valid=b['transition_valid']
            if not valid.any():continue
            current=ref(b['rgb'][valid],b['skeleton'][valid]);future=ref(b['future_rgb'][valid],b['future_skeleton'][valid])
            for j,i in enumerate(torch.where(valid)[0].tolist()):
                rows.append(dict(sequence_id=b['sequence_id'][i],frame=int(b['frame'][i]),time=float(b['time'][i]),
                                 state=current[j].cpu().tolist(),target=(future[j]-current[j]).cpu().tolist(),tau=float(b['tau'][i])))
        if not rows: raise ValueError('No valid transition prefixes in '+part)
        all_rows[part]=rows
    state=np.array([r['state'] for r in all_rows['train']]);target=np.array([r['target'] for r in all_rows['train']])
    stats=dict(state_mean=state.mean(0).tolist(),state_std=np.maximum(state.std(0),1e-6).tolist(),
               target_mean=target.mean(0).tolist(),target_std=np.maximum(target.std(0),1e-6).tolist(),
               fitted_partition='train',ddof=0,std_floor=1e-6,n_train=len(target),dimension=target.shape[1])
    save_json(out/'normalization.json',stats)
    sm,ss,tm,ts=[np.array(stats[k]) for k in ('state_mean','state_std','target_mean','target_std')]
    records={r['sequence_id']:r for r in read_jsonl(cfg['data']['manifest'])}
    for part,rows in all_rows.items():
        for r in rows:
            r['state']=((r['state']-sm)/ss).tolist();r['target']=((r['target']-tm)/ts).tolist()
            r['physical_id']=records[r['sequence_id']]['physical_id']
        write_jsonl(out/f'{part}.jsonl',rows)
    save_json(out/'reference.json',dict(config=cfg,encoder=encoder,checkpoint_sha256=file_sha256(checkpoint),
        normalization_sha256=file_sha256(out/'normalization.json'),data_fingerprints=data_fingerprints(cfg),array_fingerprints=array_fingerprints(cfg),
        split=json.loads(Path(cfg['data']['splits']).read_text()),execution=execution_provenance(cfg,checkpoint),
        synthetic=cfg['experiment']['synthetic'],partition_hashes={p:file_sha256(out/f'{p}.jsonl') for p in all_rows}))


def head(kind,dim,hidden,dropout,sigma_min):
    return nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,dim)) if kind=='deterministic' else ConditionalFutureTransition(dim,hidden,1 if kind=='single' else 4,sigma_min,dropout)


def tensors(rows):
    return torch.tensor([r['state'] for r in rows],dtype=torch.float32),torch.tensor([r['target'] for r in rows],dtype=torch.float32)


def fit(reference,kind,output,scale=1.,seed=2024,epochs=None):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Scale must be positive')
    ref=Path(reference);meta=json.loads((ref/'reference.json').read_text());cfg=meta['config'];out=fresh(output)
    for part,digest in meta['partition_hashes'].items():
        if file_sha256(ref/f'{part}.jsonl')!=digest:raise ValueError('Reference export changed')
    if file_sha256(ref/'normalization.json')!=meta['normalization_sha256']:raise ValueError('Normalization changed')
    train=read_jsonl(ref/'train.jsonl');val=read_jsonl(ref/'val.jsonl');x,y=tensors(train);vx,vy=tensors(val)
    d=y.shape[1];m=cfg['model'];t=cfg['training'];device=device_for(cfg);set_seed(seed,t['deterministic'])
    net=head(kind,d,m['head_hidden'],m['dropout'],m['sigma_min']).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    epochs=epochs or t['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=t['final_learning_rate'])
    best=None;logs=[]
    def loss(xb,yb):
        pred=net(xb.to(device));target=yb.to(device)*scale
        return nn.functional.mse_loss(pred,target) if kind=='deterministic' else -mixture_log_prob(target,pred).mean()
    for epoch in range(epochs):
        net.train();order=torch.randperm(len(x));total=0.
        for idx in order.split(t['batch_size']):
            optimizer.zero_grad();value=loss(x[idx],y[idx]);value.backward();optimizer.step();total+=float(value.detach())*len(idx)
        scheduler.step();net.eval();v=0.
        with torch.no_grad():
            for start in range(0,len(vx),t['batch_size']):
                v+=float(loss(vx[start:start+t['batch_size']],vy[start:start+t['batch_size']]))*len(vx[start:start+t['batch_size']])
        v/=len(vx);logs.append(dict(epoch=epoch,train_loss=total/len(x),validation_loss=v))
        if not np.isfinite(v):raise FloatingPointError('Nonfinite validation loss')
        if best is None or v<best:
            best=v;torch.save(dict(model=net.state_dict(),kind=kind,scale=scale,dimension=d,seed=seed,epoch=epoch,
                config=cfg,reference_sha256=file_sha256(ref/'reference.json'),normalization_sha256=meta['normalization_sha256']),out/'best.pt')
    write_jsonl(out/'training.jsonl',logs)
    save_json(out/'fit.json',dict(kind=kind,scale=scale,seed=seed,epochs=epochs,selection_partition='val',
        objective='MSE' if kind=='deterministic' else 'joint diagonal mixture NLL',reference=meta,execution=execution_provenance(cfg)))
    return out/'best.pt'


@torch.no_grad()
def predict(reference,checkpoint,partition,output):
    ref=Path(reference);ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if file_sha256(ref/'reference.json')!=ck['reference_sha256']:raise ValueError('Different reference export')
    cfg=ck['config'];m=cfg['model'];device=device_for(cfg)
    meta=json.loads((ref/'reference.json').read_text())
    if file_sha256(ref/f'{partition}.jsonl')!=meta['partition_hashes'][partition]:raise ValueError('Reference partition export changed')
    net=head(ck['kind'],ck['dimension'],m['head_hidden'],m['dropout'],m['sigma_min']).to(device);net.load_state_dict(ck['model']);net.eval()
    rows=read_jsonl(ref/f'{partition}.jsonl');result=[];scale=ck['scale']
    for start in range(0,len(rows),cfg['training']['batch_size']):
        batch=rows[start:start+cfg['training']['batch_size']];x,_=tensors(batch);p=net(x.to(device))
        for i,r in enumerate(batch):
            rec=dict(sequence_id=r['sequence_id'],frame=r['frame'])
            if ck['kind']=='deterministic':rec['mean']=(p[i]/scale).cpu().tolist()
            else:rec.update(pi=p['pi'][i].cpu().tolist(),mu=(p['mu'][i]/scale).cpu().tolist(),sigma=(p['sigma'][i]/scale).cpu().tolist())
            result.append(rec)
    if Path(output).exists():raise FileExistsError(output)
    write_jsonl(output,result)
    save_json(str(output)+'.context.json',dict(checkpoint_sha256=file_sha256(checkpoint),reference_sha256=ck['reference_sha256'],
        partition=partition,scale=scale,coordinate_system='standardized reference; inverse training scale applied before scoring',
        density_transform='mu/scale, sigma/scale; joint density includes the inverse-scale Jacobian'))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('export');a.add_argument('--config',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True);a.add_argument('--encoder',choices=['online','target'],default='target')
    a=sub.add_parser('fit');a.add_argument('--reference',required=True);a.add_argument('--kind',choices=['deterministic','single','mixture'],required=True);a.add_argument('--output',required=True);a.add_argument('--scale',type=float,default=1.);a.add_argument('--seed',type=int,default=2024);a.add_argument('--epochs',type=int)
    a=sub.add_parser('predict');a.add_argument('--reference',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--partition',choices=['val','test'],default='test');a.add_argument('--output',required=True)
    a=p.parse_args();kw=vars(a);command=kw.pop('command')
    if command=='export':kw['cfg']=load_config(kw.pop('config'))
    globals()[command](**kw)
