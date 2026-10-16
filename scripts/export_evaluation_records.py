"""Export reviewer-level run/stream/event tables from actual local evaluator outputs."""
import argparse,csv,json
from pathlib import Path


def write_csv(path,columns,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def build(runs,output,partition='test'):
    out=Path(output);run_rows=[];stream_rows=[];event_rows=[];alarm_rows=[];seen=set()
    for entry in runs:
        run=Path(entry)
        if not (run/f'{partition}_metrics.json').exists() or not (run/f'{partition}_events.jsonl').exists():
            raise FileNotFoundError(f'missing real evaluator outputs in {run}')
        metrics=json.loads((run/f'{partition}_metrics.json').read_text(encoding='utf-8'))
        rows=[json.loads(s) for s in (run/f'{partition}_events.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        seed=metrics.get('seed');dataset=metrics.get('dataset') or ('unknown')
        identity=(str(run.resolve()),partition)
        if identity in seen:raise ValueError('duplicate run folder')
        seen.add(identity)
        if metrics.get('synthetic') is True:raise ValueError('synthetic runtime-check outputs are excluded from research export')
        record=dict(run_directory=run.name,dataset=dataset,method=metrics.get('method'),seed=seed,partition=partition,
          event_f1=metrics.get('event_f1'),tp=metrics.get('tp'),fp=metrics.get('fp'),fn=metrics.get('fn'),
          adl_false_alarms=metrics.get('adl_false_alarms'),adl_hours=metrics.get('adl_hours'),
          excluded_adl_seconds=metrics.get('excluded_adl_seconds'),far=metrics.get('far'),
          completion_alt_s=metrics.get('alt'),threshold=metrics.get('threshold'))
        run_rows.append(record)
        for row in rows:
            identity2=dict(run_directory=run.name,dataset=dataset,method=metrics.get('method'),seed=seed,partition=partition,
              sequence_id=row['sequence_id'],physical_id=row['physical_id'],subject_id=row['subject_id'],
              camera=row['camera'],activity=row['activity'],tp=row['tp'],fp=row['fp'],fn=row['fn'],
              adl_false_alarms=row['adl_false_alarms'],adl_hours=row['adl_hours'],
              excluded_adl_seconds=row['excluded_adl_seconds'],valid_positions=row['valid_positions'],threshold=row['threshold'],
              valid_intervals=json.dumps(row.get('valid_intervals')),excluded_intervals_by_reason=json.dumps(row.get('excluded_intervals_by_reason')),
              far=row['adl_false_alarms']/row['adl_hours'] if row['adl_hours'] else None)
            stream_rows.append(identity2)
            matched_onsets={m['alarm_seconds'] for m in row['matches'] if m['alarm_seconds'] is not None}
            for alarm_index,onset in enumerate(row['alarm_onsets']):
                detail=next((a for a in row.get('confirmed_alarms',[]) if a['onset_seconds']==onset),{})
                alarm_rows.append({**{k:identity2[k] for k in ('run_directory','dataset','method','seed','partition','sequence_id','physical_id','subject_id','camera')},
                   **detail,
                   'alarm_index_within_stream':alarm_index,'alarm_onset_seconds':onset,
                   'is_matched_to_fall':onset in matched_onsets,'is_adl_stream':not bool(row['matches'])})
            for i,m in enumerate(row['matches']):
                detail=row.get('event_outcomes',[{} for _ in row['matches']])[i]
                event_rows.append({**{k:identity2[k] for k in ('run_directory','dataset','method','seed','partition','sequence_id','physical_id','subject_id','camera')},
                   **detail,
                   'fall_index_within_stream':i,'completion_seconds':m['completion_seconds'],
                   'impact_seconds':m['impact_seconds'],'matched_alarm_onset_seconds':m['alarm_seconds'],
                   'completion_lead_s':m['completion_lead'],'impact_lead_s':m['impact_lead'],
                   'missed':m['alarm_seconds'] is None})
    write_csv(out/'run_metrics.csv',list(run_rows[0]),run_rows)
    if stream_rows:write_csv(out/'stream_monitoring.csv',list(stream_rows[0]),stream_rows)
    if event_rows:write_csv(out/'fall_event_outcomes.csv',list(event_rows[0]),event_rows)
    if alarm_rows:write_csv(out/'confirmed_alarm_events.csv',list(alarm_rows[0]),alarm_rows)
    groups={}
    for r in stream_rows:
        key=tuple(r[k] for k in ('dataset','method','seed','subject_id','camera'))
        g=groups.setdefault(key,{k:r[k] for k in ('dataset','method','seed','subject_id','camera')})
        for field in ('adl_false_alarms','adl_hours','excluded_adl_seconds'):
            g[field]=g.get(field,0)+r[field]
    for g in groups.values():g['far']=g['adl_false_alarms']/g['adl_hours'] if g['adl_hours'] else None
    if groups:write_csv(out/'false_alarm_groups.csv',list(next(iter(groups.values()))),list(groups.values()))
    status=dict(origin='generated_from_local_evaluation_outputs',partition=partition,runs=len(run_rows),
                stream_rows=len(stream_rows),fall_event_rows=len(event_rows),confirmed_alarm_rows=len(alarm_rows),
                individual_test_records_are_restricted_to_supplied_runs=True)
    (out/'export_metadata.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    return status

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True,nargs='+');ap.add_argument('--output',required=True)
    ap.add_argument('--partition',choices=['val','test'],default='test');a=ap.parse_args()
    print(json.dumps(build(a.runs,a.output,a.partition),indent=2))
