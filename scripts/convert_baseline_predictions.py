import _common
import argparse
import csv
import json
import math
from pathlib import Path

from pft_et.utils.io import write_jsonl


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() in ('.jsonl', '.json'):
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = json.loads(path.read_text())
        if not isinstance(rows,list): raise ValueError('JSON input must contain a list of prediction records')
        return rows
    raise ValueError('Input must be CSV, JSON, or JSONL')


def convert(rows,sequence_field,frame_field,time_field,score_field,score_type,fps):
    output=[];seen=set()
    for number,row in enumerate(rows,1):
        try:
            sequence_id=str(row[sequence_field]);frame=int(row[frame_field])
            time=frame/fps if time_field is None else float(row[time_field])
            score=float(row[score_field])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f'Invalid prediction fields in row {number}') from exc
        if score_type=='logit':
            z=math.exp(-abs(score));score=1/(1+z) if score>=0 else z/(1+z)
        if not math.isfinite(score) or not 0<=score<=1:
            raise ValueError(f'Risk outside [0,1] in row {number}')
        if abs(time-frame/fps)>1e-7:
            raise ValueError(f'Time/frame mismatch in row {number}')
        key=(sequence_id,frame)
        if key in seen: raise ValueError(f'Duplicate prediction position in row {number}')
        seen.add(key)
        output.append(dict(sequence_id=sequence_id,frame=frame,time=time,risk=score))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Convert scalar baseline outputs to the PFT-ET external prediction interface.')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sequence-field',default='sequence_id');p.add_argument('--frame-field',default='frame')
    p.add_argument('--time-field',default='time',help="Use 'none' to derive time from frame/fps")
    p.add_argument('--score-field',default='risk');p.add_argument('--score-type',choices=['probability','logit'],default='probability')
    p.add_argument('--fps',type=float,default=20.)
    a=p.parse_args();time_field=None if a.time_field.lower()=='none' else a.time_field
    converted=convert(read_rows(a.input),a.sequence_field,a.frame_field,time_field,a.score_field,a.score_type,a.fps)
    write_jsonl(a.output,converted);print(json.dumps(dict(records=len(converted),output=a.output),indent=2))
