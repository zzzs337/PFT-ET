import _common
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from pft_et.utils.io import load_config,save_json,write_jsonl
from prepare_data import prepare
from pft_et.training.trainer import train
from evaluate import evaluate
from inference import inference
from aggregate_results import aggregate

if __name__=='__main__':
    from PIL import Image
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/smoke');a=p.parse_args()
    root=Path(a.output).resolve()
    if root.exists() and any(root.iterdir()): raise FileExistsError('Choose a new smoke output directory')
    torch.set_num_threads(2);root.mkdir(parents=True,exist_ok=True)
    cfg=load_config('configs/default.yaml');cfg['experiment'].update(synthetic=True,name='synthetic_smoke',output_dir=str(root/'run'),seeds=[2024])
    cfg['model'].update(rgb_backend='smoke_cnn',dim=16,skeleton_ffn=32,head_hidden=16,skeleton_layers=2)
    cfg['training'].update(epochs=1,batch_size=16,device='cpu');cfg['data']['image_size']=16
    raw=root/'raw';raw.mkdir();records=[];splits=dict(train=[],val=[],test=[])
    rng=np.random.default_rng(7)
    for part in splits:
        for fall in [True,False]:
            sid=f'{part}_{"fall" if fall else "adl"}';n=130
            folder=raw/sid;folder.mkdir()
            paths=[]
            for i in range(n):
                image=rng.integers(0,256,(16,16,3),dtype=np.uint8)
                relative=f'{sid}/{i:04d}.png';Image.fromarray(image).save(raw/relative);paths.append(relative)
            (folder/'frames.txt').write_text('\n'.join(paths));np.savetxt(folder/'timestamps.csv',np.arange(n)/20,delimiter=',')
            pose=rng.random((n,17,3)).astype(np.float32);pose[...,:2]*=16;np.save(folder/'pose.npy',pose)
            records.append(dict(sequence_id=sid,physical_id=sid,subject_id=sid,camera='1',activity='fall' if fall else 'Sitting',fall=fall,
                frame_list=f'{sid}/frames.txt',timestamps=f'{sid}/timestamps.csv',skeleton=f'{sid}/pose.npy',
                onset_seconds=3.0 if fall else None,impact_seconds=4.0 if fall else None,completion_seconds=4.5 if fall else None))
            splits[part].append(sid)
    write_jsonl(raw/'manifest.jsonl',records)
    cfg['data']['manifest']=str(prepare(raw/'manifest.jsonl',root/'processed',20,16))
    save_json(root/'splits.json',splits);cfg['data']['splits']=str(root/'splits.json');save_json(root/'config.json',cfg)
    checkpoint=train(cfg)
    evaluate(cfg,checkpoint,analysis=True)
    inference(cfg,checkpoint,cfg['data']['manifest'],root/'inference.jsonl')
    aggregate([root/'run'],root/'summary',bootstrap=100)
    save_json(root/'verification.json',dict(synthetic=True,steps=['raw_data','resampling','pose_normalization','split_check','prefix_loading','forward','three_losses','backward','optimizer_step','ema','validation_threshold','checkpoint_save_load','test_metrics','impact_recall','inference','aggregation','cluster_bootstrap']))
    print(f'Synthetic execution chain passed: {root}')
