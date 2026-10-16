import numpy as np
import pytest
from pft_et.data.dataset import PrefixDataset,label_at,make_loader
from pft_et.data.preprocessing import resampling_indices,uniform_indices,normalize_skeleton
from pft_et.data.splits import validate_splits

def test_bin_edges():
    edges=[0,10,20,30,40,50,60]
    assert label_at(0,10,True,edges)==0
    assert label_at(0,11,True,edges)==1
    assert label_at(0,60,True,edges)==5
    assert label_at(0,61,True,edges)==6
    assert label_at(0,-1,False,edges)==6
    with pytest.raises(ValueError): label_at(10,10,True,edges)

def test_future_mask_and_shapes(cfg):
    ds=PrefixDataset(cfg,'train',include_future=True)
    assert len(ds)==51+31
    batch=next(iter(make_loader(ds,cfg)))
    assert batch['rgb'].shape==(4,3,8,16,16)
    assert batch['skeleton'].shape==(4,8,51)
    assert batch['future_rgb'].shape==batch['rgb'].shape
    # Complete future can cross lying-state completion.
    item=ds[50]; assert item['frame']==89 and item['transition_valid']
    plain=PrefixDataset(cfg,'test')
    assert 'future_rgb' not in plain[0]

def test_resample_and_pose():
    grid,idx=resampling_indices([10.,10.1,10.2],20)
    assert len(grid)==5 and list(idx)==[0,0,1,1,2]
    pose=np.ones((1,17,3),dtype=np.float32);out=normalize_skeleton(pose,2,4)
    assert out[0,0].tolist()==[.5,.25,1.]
    assert len(uniform_indices(30,8))==8

def test_leakage():
    rows=[dict(sequence_id='a',subject_id='same',physical_id='p'),dict(sequence_id='b',subject_id='same',physical_id='p')]
    with pytest.raises(ValueError,match='leakage'):
        validate_splits(rows,dict(train=['a'],val=['b'],test=[]),'upfall',False)


def test_event_prefix_retained_without_future(cfg):
    from pft_et.utils.io import read_jsonl,write_jsonl
    rows=read_jsonl(cfg['data']['manifest'])
    for r in rows:
        if r['fall']:
            r['frames']=100
            for key in ('rgb_path','skeleton_path'):
                path=__import__('pathlib').Path(cfg['data']['manifest']).parent/r[key]
                np.save(path,np.load(path)[:100].copy())
    write_jsonl(cfg['data']['manifest'],rows)
    ds=PrefixDataset(cfg,'train',include_future=True)
    last=ds[50]
    assert last['frame']==89 and last['label']==0 and not last['transition_valid']

def test_causal_resampling():
    _,idx=resampling_indices([0.,.07,.19],20)
    assert list(idx)==[0,0,1,1]
