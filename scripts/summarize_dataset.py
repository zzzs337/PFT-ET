"""Export dataset counts, split membership, and physical-event/camera mappings."""
import _common
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pft_et.data.splits import validate_splits
from pft_et.utils.io import read_jsonl


def write_csv(path, fields, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def summarize(manifest, splits, dataset, output, check_counts=False):
    records=read_jsonl(manifest);split=json.loads(Path(splits).read_text(encoding='utf-8'))
    validate_splits(records,split,dataset,check_counts=check_counts)
    partition={sid:name for name in ('train','val','test') for sid in split[name]}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    membership=[]
    for r in sorted(records,key=lambda x:x['sequence_id']):
        membership.append(dict(partition=partition[r['sequence_id']],sequence_id=r['sequence_id'],physical_id=r['physical_id'],
            subject_id=r['subject_id'],camera=r['camera'],activity=r['activity'],fall=bool(r['fall']),event_id=r.get('event_id','')))
    write_csv(out/'split_membership.csv',['partition','sequence_id','physical_id','subject_id','camera','activity','fall','event_id'],membership)
    groups=defaultdict(list)
    for r in records: groups[(partition[r['sequence_id']],bool(r['fall']))].append(r)
    statistics=[]
    for part in ('train','val','test'):
        for fall in (True,False):
            rows=groups[(part,fall)]
            seconds=sum(float(r['frames'])/float(r['fps']) for r in rows if r.get('frames') is not None and r.get('fps'))
            statistics.append(dict(partition=part,class_name='fall' if fall else 'adl',camera_streams=len(rows),
                physical_sequences=len({r['physical_id'] for r in rows}),subjects=len({r['subject_id'] for r in rows}),duration_seconds=round(seconds,6)))
    write_csv(out/'dataset_statistics.csv',['partition','class_name','camera_streams','physical_sequences','subjects','duration_seconds'],statistics)
    event_rows=[]
    for r in sorted((x for x in records if x['fall']),key=lambda x:(x['physical_id'],str(x['camera']))):
        event_rows.append(dict(partition=partition[r['sequence_id']],event_id=r.get('event_id',r['physical_id']),physical_id=r['physical_id'],
            sequence_id=r['sequence_id'],subject_id=r['subject_id'],camera=r['camera'],onset_seconds=r.get('onset_seconds'),
            impact_seconds=r.get('impact_seconds'),completion_seconds=r.get('completion_seconds'),annotation_source=r.get('annotation_source',''),
            impact_annotation_source=r.get('impact_annotation_source','')))
    write_csv(out/'event_camera_map.csv',['partition','event_id','physical_id','sequence_id','subject_id','camera','onset_seconds','impact_seconds','completion_seconds','annotation_source','impact_annotation_source'],event_rows)
    report={'dataset':dataset,'streams':len(records),'physical_sequences':len({r['physical_id'] for r in records}),
        'subjects':len({r['subject_id'] for r in records}),'fall_streams':sum(bool(r['fall']) for r in records),
        'fall_physical_events':len({r['physical_id'] for r in records if r['fall']}),'protocol_counts_checked':bool(check_counts)}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--splits',required=True)
    parser.add_argument('--dataset',choices=['upfall','urfall'],required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--check-protocol-counts',action='store_true');args=parser.parse_args()
    print(json.dumps(summarize(args.manifest,args.splits,args.dataset,args.output,args.check_protocol_counts),indent=2))
