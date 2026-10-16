import _common
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pft_et.metrics.statistics import mean_sd,cluster_bootstrap
from pft_et.utils.io import read_jsonl,save_json

def aggregate(directories,output,partition='test',bootstrap=0):
    if not directories or len(set(map(str,directories)))!=len(directories):raise ValueError('Empty or duplicate run list')
    summaries=[];events=[]
    for directory in directories:
        directory=Path(directory)
        summary=json.loads((directory/f'{partition}_metrics.json').read_text())
        summaries.append(summary);events.append(read_jsonl(directory/f'{partition}_events.jsonl'))
    if len({s['synthetic'] for s in summaries})>1: raise ValueError('Cannot combine synthetic checks with research outputs')
    for field in ('dataset','method'):
        if len({s.get(field) for s in summaries})>1: raise ValueError('Cannot aggregate different '+field)
    if len({s['seed'] for s in summaries})!=len(summaries):raise ValueError('Duplicate seed in aggregation')
    signatures=[{(r['sequence_id'],r['physical_id']) for r in group} for group in events]
    if any(s!=signatures[0] for s in signatures):raise ValueError('Different evaluated stream membership')
    metrics=['event_f1','alt','recall','event_time_nll','remaining_time_mae','overall_candidate_fpr',
             'preimpact_recall@0','preimpact_recall@0.25','preimpact_recall@0.5','preimpact_recall@1']
    table=[]
    for k in metrics: table.append(dict(metric=k,**mean_sd([s[k] for s in summaries])))
    total_hours=sum(s['adl_hours'] for s in summaries);counts=sum(s['adl_false_alarms'] for s in summaries)
    rate=counts/total_hours if total_hours else None
    table.append(dict(metric='far',mean=rate,sd=mean_sd([s['far'] for s in summaries])['sd'],n=len(summaries)))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','mean','sd','n']);writer.writeheader();writer.writerows(table)
    result=dict(run_count=len(summaries),metrics=table,synthetic=summaries[0]['synthetic'],
                far_aggregation='exposure_weighted_mean_across_runs',
                conditional_poisson_95ci_by_run=[s['far_conditional_poisson_95ci'] for s in summaries],
                adl_exposure_hours_by_run=[s['adl_hours'] for s in summaries],
                adl_counts_by_run=[s['adl_false_alarms'] for s in summaries])
    if bootstrap: result['cluster_bootstrap_95ci']=cluster_bootstrap(events,bootstrap)
    for section in ['stages','candidate_fpr_by_activity']:
        result[section]=[s[section] for s in summaries]
    result['transition_analysis']=[s['transition_analysis'] for s in summaries]
    result['calibration']={k:mean_sd([s['calibration'][k] for s in summaries]) for k in ('brier_score','ece')}
    result['recall_lead_curve']=summaries[0]['recall_lead_curve']
    for i,row in enumerate(result['recall_lead_curve']):
        for key in ['completion_recall','impact_recall']:
            row[key]=mean_sd([s['recall_lead_curve'][i][key] for s in summaries])['mean']
    save_json(out/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--partition',default='test',choices=['val','test']);p.add_argument('--bootstrap',type=int,default=0)
    a=p.parse_args();aggregate(a.runs,a.output,a.partition,a.bootstrap)
