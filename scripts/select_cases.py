"""Cross-physical-sequence nearest neighbors ranked without event predictions."""
import _common
import argparse
import json
from pathlib import Path
import numpy as np
from pft_et.utils.io import read_jsonl,write_jsonl,save_json,file_sha256


def select(rows,neighbors=1):
    if neighbors<1:raise ValueError('neighbors must be positive')
    if len({(r['sequence_id'],r['frame']) for r in rows})!=len(rows):raise ValueError('Duplicate case prefix')
    x=np.array([r['state'] for r in rows]);y=np.array([r['target'] for r in rows])
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid case features')
    candidates=[];seen=set()
    for i,r in enumerate(rows):
        distance=np.sqrt(((x-x[i])**2).mean(1))
        eligible=[j for j,s in enumerate(rows) if s['physical_id']!=r['physical_id']]
        for j in sorted(eligible,key=lambda j:(distance[j],rows[j]['sequence_id'],rows[j]['frame']))[:neighbors]:
            pair=tuple(sorted((i,j)))
            if pair in seen:continue
            seen.add(pair)
            candidates.append(dict(prefix_1=dict(sequence_id=r['sequence_id'],frame=r['frame'],time=r['time']),
                prefix_2={k:rows[j][k] for k in ('sequence_id','frame','time')},
                current_distance=float(distance[j]),future_distance=float(np.sqrt(((y[i]-y[j])**2).mean()))))
    return sorted(candidates,key=lambda r:(-r['future_distance'],r['current_distance'],r['prefix_1']['sequence_id'],r['prefix_1']['frame']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--neighbors',type=int,default=1);p.add_argument('--top',type=int,default=1);p.add_argument('--predictions');p.add_argument('--candidates',help='Optional JSONL of sequence_id/frame defining the prespecified candidate pool');a=p.parse_args()
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose new case output')
    root.mkdir(parents=True,exist_ok=True);rows=read_jsonl(a.features)
    if a.top<1:raise ValueError('top must be positive')
    if a.candidates:
        allowed={(r['sequence_id'],r['frame']) for r in read_jsonl(a.candidates)}
        if not allowed<={(r['sequence_id'],r['frame']) for r in rows}:raise ValueError('Unknown candidate IDs')
        rows=[r for r in rows if (r['sequence_id'],r['frame']) in allowed]
    write_jsonl(root/'candidate_prefixes.jsonl',[{k:r[k] for k in ('sequence_id','physical_id','frame','time')} for r in rows])
    candidates=select(rows,a.neighbors)
    selected=candidates[:a.top];write_jsonl(root/'candidate_pairs.jsonl',candidates)
    if a.predictions:
        predictions={(r['sequence_id'],r['frame']):r for r in read_jsonl(a.predictions)}
        for r in selected:
            for key in ('prefix_1','prefix_2'):
                ident=r[key];prediction=predictions[(ident['sequence_id'],ident['frame'])]
                r[key]['prediction']=prediction
    save_json(root/'selected.json',dict(selected=selected,candidate_prefixes=len(rows),candidate_pairs=len(candidates),
        feature_sha256=file_sha256(a.features),candidate_pool_sha256=file_sha256(root/'candidate_prefixes.jsonl'),neighbors=a.neighbors,distance='RMS Euclidean distance in training-standardized reference coordinates',
        selection='cross-physical-sequence nearest neighbors by current state; descending observed future-transition distance',
        predictions_used_for_selection=False))
