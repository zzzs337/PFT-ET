"""Convert a reviewed annotation CSV into the raw JSONL manifest contract."""
import _common
import argparse
import csv
import json
from pathlib import Path

from pft_et.utils.io import write_jsonl


REQUIRED = ('sequence_id','physical_id','subject_id','camera','activity','fall','timestamps','skeleton')
OPTIONAL_TEXT = ('event_id','frame_list','video','bboxes','annotation_source','annotation_version','impact_annotation_source')
EVENT_FIELDS = ('onset_seconds','impact_seconds','completion_seconds')


def boolean(value, row_number):
    value=str(value).strip().lower()
    if value in ('true','1','yes'): return True
    if value in ('false','0','no'): return False
    raise ValueError(f'row {row_number}: fall must be true or false')


def relative_path(value, field, row_number):
    if not value: return None
    path=Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'row {row_number}: {field} must be manifest-relative')
    return path.as_posix()


def convert(path, check_paths=False):
    path=Path(path);root=path.resolve().parent;records=[];seen=set();physical={}
    with path.open(newline='',encoding='utf-8-sig') as stream:
        for number,row in enumerate(csv.DictReader(stream),2):
            if any(not str(row.get(field,'')).strip() for field in REQUIRED):
                raise ValueError(f'row {number}: missing required field')
            record={field:str(row[field]).strip() for field in REQUIRED if field!='fall'}
            record['fall']=boolean(row['fall'],number)
            for field in OPTIONAL_TEXT:
                value=str(row.get(field,'')).strip()
                if value: record[field]=relative_path(value,field,number) if field in ('frame_list','video','timestamps','skeleton','bboxes') else value
            for field in ('timestamps','skeleton'):
                record[field]=relative_path(record[field],field,number)
            if bool(record.get('frame_list')) == bool(record.get('video')):
                raise ValueError(f'row {number}: supply exactly one of frame_list or video')
            times=[]
            for field in EVENT_FIELDS:
                value=str(row.get(field,'')).strip()
                record[field]=None if not value else float(value)
                if record[field] is not None:
                    if record[field] < 0: raise ValueError(f'row {number}: negative event time')
                    times.append(record[field])
            if record['fall'] and record['completion_seconds'] is None:
                raise ValueError(f'row {number}: fall requires completion_seconds')
            if not record['fall'] and times:
                raise ValueError(f'row {number}: ADL event times must be empty')
            if times != sorted(times): raise ValueError(f'row {number}: event times are out of order')
            exclusions=str(row.get('excluded_intervals','')).strip()
            record['excluded_intervals']=json.loads(exclusions) if exclusions else []
            previous=0.0
            for interval in record['excluded_intervals']:
                if not isinstance(interval,list) or len(interval)!=2:
                    raise ValueError(f'row {number}: excluded_intervals must contain [start,end] pairs')
                a,b=map(float,interval)
                if a<previous or b<=a: raise ValueError(f'row {number}: excluded intervals must be ordered and disjoint')
                previous=b
            sid=record['sequence_id']
            if sid in seen or '/' in sid or '\\' in sid or '..' in sid:
                raise ValueError(f'row {number}: invalid or duplicate sequence_id')
            seen.add(sid)
            signature=(record['subject_id'],record['fall'],record['impact_seconds'],record['completion_seconds'])
            if record['physical_id'] in physical and physical[record['physical_id']]!=signature:
                raise ValueError(f'row {number}: synchronized views disagree on identity or event boundaries')
            physical[record['physical_id']]=signature
            if check_paths:
                for field in ('frame_list','video','timestamps','skeleton'):
                    if field in record and not (root/record[field]).is_file():
                        raise FileNotFoundError(f'row {number}: missing {field}: {record[field]}')
            records.append(record)
    if not records: raise ValueError('metadata table is empty')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--check-paths',action='store_true')
    args=parser.parse_args()
    records=convert(args.metadata,args.check_paths)
    write_jsonl(args.output,records)
    print(json.dumps({'records':len(records),'physical_sequences':len({r['physical_id'] for r in records}),'output':args.output},indent=2))
