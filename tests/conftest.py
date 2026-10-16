import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pft_et.utils.io import load_config,write_jsonl,save_json

@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    c=load_config('configs/default.yaml')
    c['experiment'].update(synthetic=True,output_dir=str(tmp_path/'run'))
    c['model'].update(rgb_backend='smoke_cnn',dim=16,skeleton_ffn=32,head_hidden=16)
    c['data'].update(image_size=16,manifest=str(tmp_path/'manifest.jsonl'),splits=str(tmp_path/'split.json'))
    c['training'].update(batch_size=4,epochs=1,device='cpu')
    records=[];splits=dict(train=[],val=[],test=[])
    for part in splits:
        for fall in [True,False]:
            sid=f'{part}_{fall}';folder=tmp_path/sid;folder.mkdir();n=130
            np.save(folder/'rgb.npy',np.zeros((n,16,16,3),dtype=np.uint8))
            np.save(folder/'skeleton.npy',np.zeros((n,17,3),dtype=np.float32))
            records.append(dict(sequence_id=sid,subject_id=sid,physical_id=sid,camera='1',activity='fall' if fall else 'Sitting',fall=fall,
                frames=n,fps=20,rgb_path=f'{sid}/rgb.npy',skeleton_path=f'{sid}/skeleton.npy',completion_seconds=4.5 if fall else None,impact_seconds=4.0 if fall else None))
            splits[part].append(sid)
    write_jsonl(c['data']['manifest'],records);save_json(c['data']['splits'],splits)
    return c
