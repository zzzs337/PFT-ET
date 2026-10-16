import copy
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from pft_et.data.integrity import validate_records
from pft_et.data.splits import validate_splits
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.metrics.alarms import alarm_events,match_events
from pft_et.metrics.audit import stream_audit,calibration
from pft_et.metrics.evaluation import evaluate_predictions,select_threshold
from pft_et.metrics.components import component_statistics
from pft_et.models.model import PFTET
from pft_et.utils.io import read_jsonl,save_json
from pft_et.utils.io import load_config

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_json_config_scientific_numbers(cfg,tmp_path):
    path=tmp_path/'config.json';save_json(path,cfg)
    assert load_config(path)['training']['final_learning_rate']==cfg['training']['final_learning_rate']


def test_duplicate_split_rejected(cfg):
    records=read_jsonl(cfg['data']['manifest']);split=json.loads(Path(cfg['data']['splits']).read_text());split['train'].append(split['train'][0])
    with pytest.raises(ValueError,match='Duplicate'):validate_splits(records,split,'upfall',False)


@pytest.mark.parametrize('intervals',[[[2,1]],[[1,3],[2,4]],[[0,float('nan')]]])
def test_invalid_exclusions(cfg,intervals):
    records=read_jsonl(cfg['data']['manifest']);records[0]['excluded_intervals']=intervals
    with pytest.raises(ValueError):validate_records(records)


def test_grid_and_mapping(cfg):
    records=read_jsonl(cfg['data']['manifest']);records[0]['impact_seconds']=4.01
    with pytest.raises(ValueError,match='grid'):validate_records(records)
    records=read_jsonl(cfg['data']['manifest']);records[1]['physical_id']=records[0]['physical_id']
    with pytest.raises(ValueError,match='mapping'):validate_records(records)


def test_matching_window_and_reset():
    onsets,_,_=alarm_events([0,.05,.1,.15,.2,.25,.3,.35],[1,1,1,0,0,0,1,1],.5,3,5,20)
    assert onsets==[.1]
    m,fp=match_events([1.999,2.,4.999,5.],[dict(completion_seconds=5.,impact_seconds=4.)],3.)
    assert m[0]['alarm_seconds']==2. and fp==[1.999,4.999,5.]
    with pytest.raises(ValueError):match_events([2.,2.],[],3.)


def test_audit_exposure_reset_and_missed(cfg):
    record=read_jsonl(cfg['data']['manifest'])[0]
    rows=[dict(sequence_id=record['sequence_id'],frame=f,time=f/20,risk=1.) for f in (39,40,41,45,46,47)]
    result,streams=evaluate_predictions(rows,[record],cfg,.5)
    detail=stream_audit(streams[0],rows,record,cfg)
    assert len(detail['confirmed_alarms'])==2
    assert detail['confirmed_alarms'][0]['reset_reason']=='invalid_position_gap'
    assert detail['confirmed_alarms'][0]['reset_seconds']==pytest.approx(2.1)
    assert detail['valid_duration_seconds']==.3
    covered=sum(x['end_frame_exclusive']-x['start_frame'] for x in detail['valid_intervals'])
    excluded=sum(x['end_frame_exclusive']-x['start_frame'] for g in detail['excluded_intervals_by_reason'].values() for x in g)
    assert covered+excluded==record['frames']


def test_calibration_consistency():
    rows=[dict(sequence_id='a',frame=0,label=0,probability=[1,0,0,0,0,0,0]),dict(sequence_id='b',frame=0,label=6,risk=0.)]
    assert calibration(rows)['brier_score']==0
    rows[0]['risk']=0.
    with pytest.raises(ValueError):calibration(rows)


def test_threshold_records_all_candidates(cfg):
    record=read_jsonl(cfg['data']['manifest'])[1]
    rows=[dict(sequence_id=record['sequence_id'],frame=39,time=1.95,risk=.5)]
    selected=select_threshold(rows,[record],cfg,far_budget=0)
    assert len(selected['threshold_selection']['candidates'])==99
    assert selected['threshold_selection']['partition']=='val'
    with pytest.raises(ValueError):select_threshold(rows,[record],cfg,far_budget=-1)


def test_component_statistics():
    a=component_statistics([.5,.5],[[0,0],[0,0]],[[1,1],[1,1]],[[.2,.8],[.2,.8]])
    assert a['component_overlap']==pytest.approx(1)
    assert a['component_separation']==0 and a['event_head_js']==0
    b=component_statistics([.5,.5],[[0],[2]],[[1],[1]],[[1,0],[0,1]])
    assert b['component_overlap']==pytest.approx(np.exp(-.5))
    assert b['component_separation']==2 and b['event_head_js']>0


def test_mixture_mean_not_component_means(cfg):
    model=PFTET(cfg).eval();z=torch.ones(2,cfg['model']['dim']);out=model.from_state(z,intervention='mixture_mean')
    dist=model.transition(z);mean=(dist['pi'][...,None]*dist['mu']).sum(1)
    expected=model.event_time.net(torch.cat([z,mean],-1)).softmax(-1)
    torch.testing.assert_close(out['probability'],expected)


def test_cross_sequence_donors():
    from run_interventions import donor_indices
    ids=['a','a','b','b','c'];donors=donor_indices(ids,7)
    assert donors==donor_indices(ids,7)
    assert all(ids[i]!=ids[j] for i,j in enumerate(donors))
    with pytest.raises(ValueError):donor_indices(['a','a'],7)


def test_case_selection_ignores_predictions():
    from select_cases import select
    rows=[dict(sequence_id=str(i),physical_id=str(i//2),frame=0,time=0.,state=[float(i)],target=[float(i*i)]) for i in range(4)]
    result=select(rows)
    for r in rows:r['probability']=[0,1]
    assert select(rows)==result
    assert all(int(r['prefix_1']['sequence_id'])//2!=int(r['prefix_2']['sequence_id'])//2 for r in result)


def test_scale_inverse_nll():
    from evaluate_shared_transition import calculate
    target={('s',0):dict(target=[.2,-.3])}
    native=calculate(target,{('s',0):dict(pi=[1.],mu=[[.1,.1]],sigma=[[.7,.7]])})
    for scale in (.5,2.):
        mu=np.array([[.1,.1]])*scale;sd=np.array([[.7,.7]])*scale
        back=calculate(target,{('s',0):dict(pi=[1.],mu=(mu/scale).tolist(),sigma=(sd/scale).tolist())})
        assert back['transition_nll']==pytest.approx(native['transition_nll'])


def test_evaluation_roundtrip_and_tamper(cfg,tmp_path):
    from pft_et.training.trainer import train
    from evaluate import evaluate
    from recompute_evaluation import recompute
    ck=train(cfg);out=tmp_path/'evaluation'
    evaluate(cfg,ck,output=out,analysis=True)
    assert recompute(out)['status']=='PASS'
    with pytest.raises(FileExistsError):evaluate(cfg,ck,output=out)
    path=out/'test_metrics.json';value=json.loads(path.read_text());value['tp']+=1;save_json(path,value)
    assert recompute(out)['status']=='FAIL'


def test_baseline_search_adapter_contract(cfg,tmp_path):
    from run_baseline_search import execute
    adapter=tmp_path/'fixture_adapter.py'
    adapter.write_text('''from pathlib import Path
from pft_et.utils.io import read_jsonl,write_jsonl
from pft_et.data.integrity import valid_frames
import json
def fit(config,params,seed,output):
    # Generated-input contract fixture; not a baseline model or research result.
    folder=Path(output);folder.mkdir(parents=True);path=folder/'fixture.bin';path.write_bytes(b'contract fixture');return str(path)
def predict(config,checkpoint,partition,output):
    split=json.loads(Path(config['data']['splits']).read_text());allowed=set(split[partition]);rows=[]
    for r in read_jsonl(config['data']['manifest']):
        if r['sequence_id'] in allowed:
            rows.extend(dict(sequence_id=r['sequence_id'],frame=f,time=f/config['data']['fps'],risk=0.) for f in valid_frames(r,config))
    write_jsonl(output,rows);return output
''')
    folder=tmp_path/'search';execute(cfg,'tfd',str(adapter),folder)
    selected=json.loads((folder/'selected.json').read_text())
    assert selected['selection_partition']=='val' and selected['test_used'] is False
    assert len(read_jsonl(folder/'validation_records.jsonl'))==6
    masked=json.loads((folder/'trial000/search_inputs/splits.json').read_text())
    assert not masked['test']


def test_array_contents_fingerprinted(cfg):
    from pft_et.utils.io import array_fingerprints
    before=array_fingerprints(cfg);r=read_jsonl(cfg['data']['manifest'])[0]
    path=Path(cfg['data']['manifest']).parent/r['skeleton_path'];pose=np.load(path);pose[0,0,0]+=1;np.save(path,pose)
    assert array_fingerprints(cfg)!=before


@pytest.mark.parametrize('field,value',[('observation_frames',41),('rgb_std',[1.,1.,1.])])
def test_checkpoint_preprocessing_identity(cfg,field,value):
    from pft_et.utils.compatibility import check_checkpoint_config
    changed=copy.deepcopy(cfg);changed['data'][field]=value
    with pytest.raises(ValueError,match='preprocessing'):check_checkpoint_config(changed,cfg)


def test_checkpoint_explicit_inference_variation(cfg):
    from pft_et.utils.compatibility import check_checkpoint_config
    changed=copy.deepcopy(cfg);changed['model']['test_samples']=16;changed['model']['marginalization']='mean'
    changed['data']['manifest']='relocated/manifest.jsonl'
    check_checkpoint_config(changed,cfg)
