"""Auditable frame-cell exposure and alarm/event records."""
import numpy as np


def frame_intervals(frames, fps):
    intervals=[]
    for f in sorted(set(int(f) for f in frames)):
        if intervals and f==intervals[-1][1]: intervals[-1][1]=f+1
        else: intervals.append([f,f+1])
    return [dict(start_frame=a,end_frame_exclusive=b,start_seconds=a/fps,end_seconds_exclusive=b/fps) for a,b in intervals]


def stream_audit(row, predictions, record, cfg):
    fps=cfg['data']['fps'];n=record['frames'];wo=cfg['data']['observation_frames']
    from pft_et.metrics.alarms import AlarmConfirmation
    from pft_et.metrics.evaluation import prediction_risk
    state=AlarmConfirmation(cfg['alarm']['k'],cfg['alarm']['w'])
    bins=cfg['data']['bin_edges'].index(cfg['alarm']['horizon_seconds']);states=len(cfg['data']['bin_edges'])
    alarms=[];active=None;trace=[];prev=None
    for p in sorted(predictions,key=lambda p:p['frame']):
        t=float(p['time']);gap=prev is not None and p['frame']!=prev+1
        if gap and active is not None:
            active.update(reset_seconds=(prev+1)/fps,reset_reason='invalid_position_gap');active=None
        was=state.confirmed
        risk=prediction_risk(p,bins,states)
        candidate,confirmed,onset=state.step(risk,row['threshold'],t,1/fps)
        if was and not confirmed and not gap and active is not None:
            active.update(reset_seconds=t,reset_reason='confirmation_ended');active=None
        if onset:
            active=dict(alarm_id=f"{row['sequence_id']}:alarm:{len(alarms)}",onset_seconds=t,
                        reset_seconds=None,reset_reason='stream_end_censored',
                        outcome='TP' if t not in row['unmatched_alarm_onsets'] else 'FP')
            alarms.append(active)
        trace.append(dict(frame=int(p['frame']),time=t,risk=risk,candidate=candidate,confirmed=confirmed,onset=onset,gap_reset=gap))
        prev=int(p['frame'])
    frames={int(p['frame']) for p in predictions};reasons={}
    for f in range(n):
        if f in frames: continue
        if f<wo-1: reason='observation_warmup'
        elif not record['fall'] and f+int(round(cfg['data']['bin_edges'][-1]*fps))>n-1: reason='unobservable_event_horizon'
        elif record['fall'] and f/fps>=record['completion_seconds']: reason='at_or_after_completion'
        else: reason='annotated_exclusion_or_missing_prediction'
        reasons.setdefault(reason,[]).append(f)
    events=[]
    for i,m in enumerate(row['matches']):
        matched=next((a for a in alarms if a['onset_seconds']==m['alarm_seconds']),None)
        event=dict(m,event_id=record.get('event_id',record['physical_id']),
                   stream_event_id=f"{row['sequence_id']}:fall:{i}",missed=m['alarm_seconds'] is None,
                   alarm_id=matched['alarm_id'] if matched else None,onset_seconds=record.get('onset_seconds'))
        for lead in (0.,.25,.5,1.):
            value=m['impact_lead']
            event[f'preimpact_success@{lead:g}']=None if m['impact_seconds'] is None else bool(value is not None and (value>1e-9 if lead==0 else value>=lead-1e-9))
        events.append(event)
    return dict(confirmed_alarms=alarms,event_outcomes=events,alarm_trace=trace,
                valid_intervals=frame_intervals(frames,fps),excluded_intervals_by_reason={k:frame_intervals(v,fps) for k,v in reasons.items()},
                annotated_excluded_intervals=record.get('excluded_intervals',[]),full_duration_seconds=n/fps,valid_duration_seconds=len(frames)/fps,
                time_definition='first confirmed alarm onset on source frame timeline; confirmation included; computation not deducted')


def calibration(rows, states=7, bins=10):
    if bins<2: raise ValueError('At least two calibration bins required')
    from pft_et.metrics.evaluation import prediction_risk
    seen=set();risk=[];target=[]
    for r in rows:
        key=(r['sequence_id'],r['frame'])
        if key in seen: raise ValueError('Duplicate calibration prefix')
        seen.add(key); label=r['label']
        if not 0<=label<states: raise ValueError('Invalid label')
        risk.append(prediction_risk(r,states-1,states));target.append(float(label!=states-1))
    if not risk: raise ValueError('Empty calibration input')
    p=np.asarray(risk);y=np.asarray(target);index=np.minimum((p*bins).astype(int),bins-1)
    result=[];ece=0.
    for i in range(bins):
        mask=index==i;n=int(mask.sum());a=float(p[mask].mean()) if n else None;b=float(y[mask].mean()) if n else None
        if n: ece+=n/len(p)*abs(a-b)
        result.append(dict(bin=i,lower=i/bins,upper=(i+1)/bins,count=n,mean_probability=a,observed_frequency=b))
    return dict(n_prefixes=len(p),positive_prefixes=int(y.sum()),brier_score=float(((p-y)**2).mean()),ece=float(ece),
                n_bins=bins,bins=result,unit_of_analysis='valid prediction prefix',bin_rule='[lower,upper); last upper inclusive')
