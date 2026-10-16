import numpy as np
import pytest
from pft_et.metrics.alarms import alarm_events,match_events
from pft_et.metrics.evaluation import preimpact_recall,prefix_metrics,evaluate_predictions,select_threshold,prediction_risk
from pft_et.metrics.statistics import poisson_interval

def test_alarm_confirmation_and_gap():
    t=np.arange(8)/20
    onsets,c,a=alarm_events(t,[1,1,1,1,1,0,0,0],.5,3,5,20)
    assert onsets==[.1]
    # The confirmation can remain active after the candidate is false.
    assert a[5] and a[6] and not a[7]
    onsets,_,_=alarm_events([0,.05,.2,.25,.3],[1]*5,.5,3,5,20)
    assert onsets==[.3]

def test_matching_boundaries_and_no_recount():
    falls=[dict(completion_seconds=5.,impact_seconds=4.)]
    matches,unmatched=match_events([1.9,2.,3.,5.],falls,3.)
    assert matches[0]['alarm_seconds']==2.
    assert unmatched==[1.9,3.,5.]
    assert matches[0]['completion_lead']==3.

def test_preimpact_recall_retains_missed_events():
    rows=[dict(impact_seconds=2.,impact_lead=.5),dict(impact_seconds=2.,impact_lead=0.),dict(impact_seconds=2.,impact_lead=None)]
    assert preimpact_recall(rows,0)==1/3
    assert preimpact_recall(rows,.5)==1/3
    assert preimpact_recall(rows,.5,-.05)==0

def test_mae_tail_normalization():
    p=np.array([[.1,0,0,0,0,0,.9]])
    m=prefix_metrics(p,np.array([0]),np.array([.25]),[0,.5,1,1.5,2,2.5,3])
    assert m['remaining_time_mae']==0
    assert abs(m['event_time_nll']+np.log(.1))<1e-10

def test_zero_false_alarm_ci():
    low,high=poisson_interval(0,1)
    assert low==0 and 3.68<high<3.70

def test_event_fp_and_far_are_distinct(cfg):
    records=[dict(sequence_id='f',physical_id='f',subject_id='s',camera='1',activity='Fall',fall=True,completion_seconds=5.,impact_seconds=4.,frames=150),
             dict(sequence_id='a',physical_id='a',subject_id='s',camera='1',activity='Walking',fall=False,frames=150)]
    cfg['alarm'].update(k=1,w=1)
    rows=[]
    for sid in ['f','a']:
        for i in range(130):
            risk = 1. if sid=='f' and i in (20,60,62) else 0.
            prob=[risk,0,0,0,0,0,1-risk]
            rows.append(dict(sequence_id=sid,time=i/20,frame=i,probability=prob,label=6,tau=-1))
    result,events=evaluate_predictions(rows,records,cfg,.5)
    assert result['tp']==1 and result['fp']==2 and result['far']==0
    assert result['event_f1']==.5

def test_scalar_risk_uses_same_event_protocol_without_time_metrics(cfg):
    records=[dict(sequence_id='f',physical_id='f',subject_id='s',camera='1',activity='Fall',fall=True,completion_seconds=5.,impact_seconds=4.,frames=150)]
    cfg['alarm'].update(k=1,w=1)
    scalar=[];distribution=[]
    for i in range(100):
        risk=1. if i==60 else 0.
        common=dict(sequence_id='f',time=i/20,frame=i,label=0,tau=(100-i)/20)
        scalar.append(dict(**common,risk=risk))
        distribution.append(dict(**common,probability=[risk,0,0,0,0,0,1-risk]))
    a,_=evaluate_predictions(scalar,records,cfg,.5);b,_=evaluate_predictions(distribution,records,cfg,.5)
    assert (a['tp'],a['fp'],a['fn'],a['event_f1'],a['alt'])==(b['tp'],b['fp'],b['fn'],b['event_f1'],b['alt'])
    assert a['event_time_nll'] is None and a['remaining_time_mae'] is None
    assert not a['event_time_metrics_available'] and b['event_time_metrics_available']

def test_risk_probability_consistency_and_bounds():
    assert prediction_risk(dict(risk=.4),6,7)==.4
    assert prediction_risk(dict(probability=[.1,.1,.1,.1,0,0,.6]),6,7)==pytest.approx(.4)
    with pytest.raises(ValueError): prediction_risk(dict(risk=1.1),6,7)
    with pytest.raises(ValueError): prediction_risk(dict(risk=.5,probability=[.1,.1,.1,.1,0,0,.6]),6,7)
