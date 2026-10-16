"""Serial online latency with raw or prepared inputs and synchronized stage timers."""
import _common
import argparse
import copy
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from pft_et.models.model import PFTET
from pft_et.metrics.alarms import AlarmConfirmation
from pft_et.data.dataset import PrefixDataset,collate_fn
from pft_et.data.preprocessing import resampling_indices,uniform_indices,resize_rgb,normalize_skeleton
from pft_et.training.prediction import to_device
from pft_et.training.trainer import device_for
from pft_et.utils.io import load_config,read_jsonl,write_jsonl,save_json,file_sha256
from pft_et.utils.checkpoint import load_checkpoint
from pft_et.utils.provenance import execution_provenance
from pft_et.utils.seed import set_seed
from pft_et.utils.compatibility import check_checkpoint_config


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


class Prepared:
    def __init__(self,cfg):
        self.dataset=PrefixDataset(cfg,'test');self.n=len(self.dataset)
    def get(self,index):return collate_fn([self.dataset[index]])


class Raw:
    def __init__(self,cfg,manifest,pose_config,pose_checkpoint,device):
        from mmpose.apis import init_model,inference_topdown
        from PIL import Image
        self.infer=inference_topdown;self.Image=Image;self.cfg=cfg
        if not pose_config or not pose_checkpoint:raise ValueError('Raw timing requires RTMPose config and weights')
        self.model=init_model(pose_config,pose_checkpoint,device=str(device));self.root=Path(manifest).resolve().parent
        rows=read_jsonl(manifest)
        if len(rows)!=1:raise ValueError('Raw timing manifest must select one complete camera stream')
        r=rows[0]
        if 'frame_list' not in r:raise ValueError('Raw timing uses a source frame_list to measure individual image reads')
        self.paths=(self.root/r['frame_list']).read_text().splitlines();ts=np.loadtxt(self.root/r['timestamps'],delimiter=',',ndmin=1)
        self.grid,self.indices=resampling_indices(ts,cfg['data']['fps']);self.n=len(self.grid)
        if len(self.paths)!=len(ts):raise ValueError('Frame/timestamp count mismatch')
        self.boxes=np.load(self.root/r['bboxes']) if r.get('bboxes') else None
        self.rgb=deque(maxlen=cfg['data']['observation_frames']);self.pose=deque(maxlen=cfg['data']['observation_frames'])
        self.sid=r['sequence_id']
    def read(self,index):
        return np.asarray(self.Image.open(self.root/self.paths[self.indices[index]]).convert('RGB'))
    def estimate_pose(self,rgb,index):
        bboxes=self.boxes[self.indices[index]:self.indices[index]+1] if self.boxes is not None else None
        result=self.infer(self.model,rgb[...,::-1].copy(),bboxes=bboxes)
        if len(result)!=1:raise ValueError('One tracked person is required')
        p=result[0].pred_instances
        return np.concatenate([np.asarray(p.keypoints)[0],np.asarray(p.keypoint_scores)[0,:,None]],axis=1)
    def prepare(self,rgb,pose,index):
        d=self.cfg['data'];self.rgb.append(resize_rgb(rgb,d['image_size']))
        self.pose.append(normalize_skeleton(pose[None],rgb.shape[1],rgb.shape[0])[0])
        if len(self.rgb)<d['observation_frames']:return None
        ids=uniform_indices(len(self.rgb),d['sampled_frames']);x=torch.from_numpy(np.array(self.rgb)[ids].copy()).float().permute(3,0,1,2)/255
        mean=torch.tensor(d['rgb_mean'])[:,None,None,None];std=torch.tensor(d['rgb_std'])[:,None,None,None]
        return dict(rgb=((x-mean)/std)[None],skeleton=torch.from_numpy(np.array(self.pose)[ids].copy()).float().flatten(1)[None],time=index/d['fps'])


@torch.no_grad()
def benchmark(cfg,checkpoint,output,mode='prepared',warmup=20,repeats=100,raw_manifest=None,pose_config=None,pose_checkpoint=None):
    if cfg['model']['strategy']=='history':raise ValueError('History strategy requires a history-buffer timing adapter')
    if warmup<0 or repeats<2:raise ValueError('Invalid timing sample counts')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new timing output')
    out.mkdir(parents=True,exist_ok=True);device=device_for(cfg);set_seed(cfg['training']['seed'],cfg['training']['deterministic'])
    ck=load_checkpoint(checkpoint,device=device);model=PFTET(cfg,initialize=False).to(device);model.load_state_dict(ck['model']);model.eval()
    check_checkpoint_config(cfg,ck['config'])
    source=Raw(cfg,raw_manifest,pose_config,pose_checkpoint,device) if mode=='raw' else Prepared(cfg)
    state=AlarmConfirmation(cfg['alarm']['k'],cfg['alarm']['w']);rows=[];valid=0
    for index in range(source.n):
        times={};synchronize(device);total_start=time.perf_counter_ns()
        def timed(key,fn):
            synchronize(device);start=time.perf_counter_ns();result=fn();synchronize(device)
            times[key]=(time.perf_counter_ns()-start)/1e6;return result
        if mode=='raw':
            rgb=timed('read_ms',lambda:source.read(index));pose=timed('pose_ms',lambda:source.estimate_pose(rgb,index))
            batch=timed('preprocess_ms',lambda:source.prepare(rgb,pose,index))
            if batch is None:continue
            timestamp=index/cfg['data']['fps']
        elif mode=='model':
            batch=source.get(index);b=to_device(batch,device);synchronize(device);total_start=time.perf_counter_ns()
            timed('model_forward_ms',lambda:model(b)['probability']);synchronize(device)
            times['total_ms']=(time.perf_counter_ns()-total_start)/1e6
            if valid>=warmup:rows.append(dict(iteration=valid-warmup,source_index=index,**times))
            valid+=1
            if len(rows)>=repeats:break
            continue
        else:
            times['pose_ms']=None;times['read_ms']=None
            batch=timed('prepared_read_and_preprocess_ms',lambda:source.get(index));timestamp=float(batch['time'][0])
        b=timed('host_to_device_ms',lambda:to_device(batch,device))
        z=timed('rgb_skeleton_encoding_fusion_ms',lambda:model.online(b['rgb'],b['skeleton']))
        prediction=timed('transition_mc_ms',lambda:model.from_state(z)['probability'])
        def confirm():
            bins=cfg['data']['bin_edges'].index(cfg['alarm']['horizon_seconds'])
            risk=float(prediction[0,:bins].sum().cpu());return state.step(risk,ck['threshold'],timestamp,1/cfg['data']['fps'])
        timed('risk_transfer_confirmation_ms',confirm);synchronize(device)
        times['total_ms']=(time.perf_counter_ns()-total_start)/1e6
        if valid>=warmup:rows.append(dict(iteration=valid-warmup,source_index=index,**times))
        valid+=1
        if len(rows)>=repeats:break
    if len(rows)<repeats:raise ValueError('Insufficient stream positions for warmup + repeats')
    write_jsonl(out/'timing.jsonl',rows)
    summary={}
    for key in rows[0]:
        if key.endswith('_ms') and rows[0][key] is not None:
            v=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
    scope={'raw':'raw frame read through confirmed alarm','prepared':'prepared-array read through confirmed alarm; excludes pose extraction and source video decoding','model':'model forward only; prepared tensors already on device; excludes input reads, transfers, pose, and confirmation'}
    save_json(out/'timing_summary.json',dict(mode=mode,scope=scope[mode],
        warmup_predictions=warmup,measured_predictions=len(rows),batch_size=1,mc_samples_per_component=cfg['model']['test_samples'],
        synchronization='torch.cuda.synchronize before/after each stage and full prediction on CUDA',timer='perf_counter_ns wall time',
        throughput_predictions_per_second=1000/np.mean([r['total_ms'] for r in rows]),summary=summary,
        cached_history='causal rolling RGB/pose buffer in raw mode',config=cfg,synthetic=cfg['experiment']['synthetic'],
        pose_checkpoint_sha256=file_sha256(pose_checkpoint) if pose_checkpoint else None,
        pose_config_sha256=file_sha256(pose_config) if pose_config else None,execution=execution_provenance(cfg,checkpoint)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['raw','prepared','model'],default='prepared');p.add_argument('--warmup',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--raw-manifest');p.add_argument('--pose-config');p.add_argument('--pose-checkpoint');a=p.parse_args();kw=vars(a);kw['cfg']=load_config(kw.pop('config'));benchmark(**kw)
