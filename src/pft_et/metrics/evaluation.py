from collections import defaultdict
import numpy as np
from pft_et.metrics.alarms import alarm_events,match_events
from pft_et.metrics.statistics import poisson_interval

def prediction_risk(prediction,bins,states):
    probability=prediction.get('probability')
    scalar=prediction.get('risk')
    derived=None
    if probability is not None:
        probability=np.asarray(probability,dtype=float)
        if len(probability)!=states or not np.isfinite(probability).all() or (probability<0).any() or abs(probability.sum()-1)>1e-5:
            raise ValueError('Invalid event-time probability vector')
        derived=float(probability[:bins].sum())
    if scalar is not None:
        scalar=float(scalar)
        if not np.isfinite(scalar) or not 0<=scalar<=1:
            raise ValueError('Invalid scalar fall-risk score')
    if probability is None and scalar is None:
        raise ValueError('Prediction requires risk or probability')
    if derived is not None and scalar is not None and abs(derived-scalar)>1e-5:
        raise ValueError('Scalar risk disagrees with event-time probability')
    return derived if scalar is None else scalar

def preimpact_recall(matches,lead,shift=0.):
    evaluable=[m for m in matches if m['impact_seconds'] is not None]
    if not evaluable: return None
    count=0
    for m in evaluable:
        if m['impact_lead'] is None: continue
        value=m['impact_lead']+shift
        count += int(value>1e-9 if lead==0 else value>=lead-1e-9)
    return count/len(evaluable)

def summarize_rows(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    count=sum(r['adl_false_alarms'] for r in rows);hours=sum(r['adl_hours'] for r in rows)
    matches=[m for r in rows for m in r['matches']]
    leads=[m['completion_lead'] for m in matches if m['completion_lead'] is not None]
    result=dict(tp=tp,fp=fp,fn=fn,event_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                recall=tp/(tp+fn) if tp+fn else None,alt=float(np.mean(leads)) if leads else None,
                adl_false_alarms=count,adl_hours=hours,far=count/hours if hours else None,
                excluded_adl_seconds=sum(r['excluded_adl_seconds'] for r in rows))
    result['far_conditional_poisson_95ci']=poisson_interval(count,hours)
    for lead in (0.,0.25,0.5,1.0):
        result[f'preimpact_recall@{lead:g}']=preimpact_recall(matches,lead)
        result[f'preimpact_recall_minus1frame@{lead:g}']=preimpact_recall(matches,lead,-0.05)
        result[f'preimpact_recall_plus1frame@{lead:g}']=preimpact_recall(matches,lead,0.05)
    # Both completion and impact curves retain missed falls in their denominator.
    result['recall_lead_curve']=[dict(lead_seconds=float(l),
         completion_recall=sum(m['completion_lead'] is not None and m['completion_lead']>=l-1e-9 for m in matches)/len(matches) if matches else None,
         impact_recall=preimpact_recall(matches,float(l))) for l in np.arange(0,3.01,0.05)]
    return result

def evaluate_predictions(predictions,records,cfg,threshold):
    if not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid threshold')
    if len({r['sequence_id'] for r in records})!=len(records): raise ValueError('Duplicate stream metadata')
    byid=defaultdict(list)
    metadata={r['sequence_id']:r for r in records}
    if {p['sequence_id'] for p in predictions}-metadata.keys(): raise ValueError('Unknown prediction stream')
    a=cfg['alarm'];fps=cfg['data']['fps'];states=len(cfg['data']['bin_edges']);bins=cfg['data']['bin_edges'].index(a['horizon_seconds'])
    output_types=set()
    for p in predictions:
        prediction_risk(p,bins,states)
        output_types.add('probability' if p.get('probability') is not None else 'risk')
        byid[p['sequence_id']].append(p)
    if len(output_types)>1: raise ValueError('Do not mix scalar-risk and event-time prediction rows')
    rows=[];all_labels=[];all_p=[];all_tau=[];all_analysis=[]
    for sid,record in metadata.items():
        seq=sorted(byid.get(sid,[]),key=lambda p:p['time'])
        times=np.array([p['time'] for p in seq]);risk=np.array([prediction_risk(p,bins,states) for p in seq])
        if not np.isfinite(times).all(): raise ValueError('Nonfinite prediction timestamp')
        if any(abs(p['frame']/fps-p['time'])>1e-7 for p in seq): raise ValueError('Frame/time mismatch')
        if len(np.unique(times))!=len(times): raise ValueError('Duplicate prediction time')
        if np.any(times<0) or np.any(times>(record['frames']-1)/fps+1e-8): raise ValueError('Prediction outside stream')
        onsets,candidates,confirmed=alarm_events(times,risk,threshold,a['k'],a['w'],fps)
        falls=[record] if record['fall'] else []
        matches,unmatched=match_events(onsets,falls,a['horizon_seconds'])
        tp=sum(m['alarm_seconds'] is not None for m in matches)
        hours=len(seq)/fps/3600 if not record['fall'] else 0.
        rows.append(dict(sequence_id=sid,physical_id=record['physical_id'],subject_id=record['subject_id'],
            camera=record['camera'],activity=record['activity'],
            cluster_id=record['subject_id'] if cfg['data']['dataset']=='upfall' else record['physical_id'],
            threshold=threshold,tp=tp,fn=len(falls)-tp,fp=len(unmatched),matches=matches,
            alarm_onsets=onsets,unmatched_alarm_onsets=unmatched,adl_false_alarms=len(onsets) if not record['fall'] else 0,
            adl_hours=hours,excluded_adl_seconds=record['frames']/fps-len(seq)/fps if not record['fall'] else 0,
            valid_positions=len(seq),candidate_positives=int(candidates.sum()) if not record['fall'] else 0,
            candidate_fpr=float(candidates.mean()) if not record['fall'] and len(seq) else None))
        for p in seq:
            if p.get('label',-1)>=0 and p.get('probability') is not None:
                all_labels.append(p['label']);all_p.append(p['probability']);all_tau.append(p.get('tau',-1))
            if p.get('transition') is not None: all_analysis.append(p)
    result=summarize_rows(rows)
    result.update(prefix_metrics(np.array(all_p),np.array(all_labels),np.array(all_tau),cfg['data']['bin_edges']))
    result['event_time_metrics_available']=bool(all_p)
    result['threshold']=threshold
    classes=defaultdict(lambda:[0,0])
    for row in rows:
        if row['adl_hours']>0:
            classes[row['activity']][0]+=row['candidate_positives'];classes[row['activity']][1]+=row['valid_positions']
    result['candidate_fpr_by_activity']={k:v[0]/v[1] for k,v in classes.items()}
    result['overall_candidate_fpr']=sum(v[0] for v in classes.values())/sum(v[1] for v in classes.values()) if classes else None
    result['transition_analysis']=transition_metrics(all_analysis)
    for field in ['cluster_id','camera','sequence_id']:
        grouped=defaultdict(list)
        for row in rows: grouped[str(row[field])].append(row)
        result[f'by_{field}']={key:summarize_rows(values) for key,values in grouped.items()}
    return result,rows

def prefix_metrics(prob,label,tau,edges):
    if not len(label): return dict(event_time_nll=None,remaining_time_mae=None,stages={})
    nll=-np.log(np.maximum(prob[np.arange(len(label)),label.astype(int)],1e-12))
    centers=(np.array(edges[:-1])+np.array(edges[1:]))/2
    estimate=(prob[:,:-1]*centers).sum(-1)/np.maximum(prob[:,:-1].sum(-1),1e-12)
    mask=(tau>0)&(tau<=edges[-1]+1e-9)
    mae=abs(estimate-tau)
    stages={}
    for name,lo,hi in [('Early',2,3),('Middle',1,2),('Near-event',0,1)]:
        m=(tau>lo+1e-9)&(tau<=hi+1e-9)
        stages[name]=dict(n=int(m.sum()),nll=float(nll[m].mean()) if m.any() else None,mae=float(mae[m].mean()) if m.any() else None)
    return dict(event_time_nll=float(nll.mean()),remaining_time_mae=float(mae[mask].mean()) if mask.any() else None,stages=stages)

def transition_metrics(predictions):
    if not predictions: return {}
    squared=[p['transition']['squared_error'] for p in predictions]
    nll=[p['transition']['nll'] for p in predictions if p['transition'].get('nll') is not None]
    stages={}
    for name,lo,hi in [('Early',2,3),('Middle',1,2),('Near-event',0,1)]:
        rows=[p for p in predictions if lo+1e-9<p.get('tau',-1)<=hi+1e-9 and 'effective_components' in p['transition']]
        stages[name]=dict(n=len(rows),effective_components=float(np.mean([r['transition']['effective_components'] for r in rows])) if rows else None,
             maximum_weight=float(np.mean([r['transition']['maximum_weight'] for r in rows])) if rows else None)
        for key in ('component_overlap','component_separation','event_head_js'):
            values=[r['transition'][key] for r in rows if r['transition'].get(key) is not None]
            stages[name][key]=float(np.mean(values)) if values else None
    return dict(rmse=float(np.sqrt(np.mean(squared))),nll=float(np.mean(nll)) if nll else None,stages=stages)

def select_threshold(predictions,records,cfg,far_budget=None):
    if far_budget is not None and (not np.isfinite(far_budget) or far_budget<0): raise ValueError('Invalid FAR budget')
    start,end,step=cfg['alarm']['thresholds'];candidates=[];trace=[]
    for threshold in np.arange(start,end+step/2,step):
        result,_=evaluate_predictions(predictions,records,cfg,float(round(threshold,8)))
        far=result['far']
        trace.append(dict({k:result[k] for k in ('threshold','event_f1','recall','far','tp','fp','fn')},
                          eligible=far_budget is None or (far is not None and far<=far_budget+1e-12)))
        if far_budget is not None and (far is None or far>far_budget+1e-12): continue
        score=result['event_f1'] if far_budget is None else (result['recall'] or 0.)
        candidates.append(((score,-(far if far is not None else float('inf')),float(threshold)),result))
    if not candidates: raise ValueError('No validation threshold satisfies the FAR budget')
    # Higher threshold is a deterministic final tie break (implementation assumption).
    selected=max(candidates,key=lambda x:x[0])[1]
    selected['threshold_selection']=dict(partition='val',far_budget=far_budget,grid=[start,end,step],
        tie_break='higher score, lower FAR, higher threshold',
        candidates=trace)
    return selected
