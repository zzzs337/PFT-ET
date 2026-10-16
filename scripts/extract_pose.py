import _common
import argparse
from pathlib import Path
import numpy as np
from pft_et.utils.io import read_jsonl
from prepare_data import read_source

def extract(manifest,config,checkpoint,device):
    from mmpose.apis import init_model,inference_topdown
    if not Path(config).is_file() or not Path(checkpoint).is_file(): raise FileNotFoundError('Supply local RTMPose-m COCO config and checkpoint')
    model=init_model(config,checkpoint,device=device)
    root=Path(manifest).resolve().parent
    for r in read_jsonl(manifest):
        frames,_=read_source(r,root);poses=[]
        boxes=np.load(root/r['bboxes'],allow_pickle=False) if r.get('bboxes') else None
        if boxes is not None and boxes.shape!=(len(frames),4): raise ValueError('bboxes must be [source_frames,4], xyxy')
        for i,rgb in enumerate(frames):
            # Full-frame single-person bbox is the configured default assumption.
            results=inference_topdown(model,rgb[...,::-1].copy(),bboxes=boxes[i:i+1] if boxes is not None else None)
            if len(results)!=1: raise ValueError('Expected one tracked subject; provide one bbox per frame')
            inst=results[0].pred_instances
            xy=np.asarray(inst.keypoints)[0];scores=np.asarray(inst.keypoint_scores)[0]
            if xy.shape!=(17,2): raise ValueError('Resource must output COCO 17 joints')
            poses.append(np.concatenate([xy,scores[:,None]],axis=1))
        output=root/r['skeleton'];output.parent.mkdir(parents=True,exist_ok=True)
        np.save(output,np.asarray(poses,dtype=np.float32))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--pose-config',required=True);p.add_argument('--pose-checkpoint',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();extract(a.manifest,a.pose_config,a.pose_checkpoint,a.device)
