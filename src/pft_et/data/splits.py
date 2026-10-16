import json
from collections import Counter
import numpy as np
from pft_et.utils.io import save_json

def validate_splits(records, split, dataset, check_counts=True):
    ids = [r['sequence_id'] for r in records]
    for name in ('train','val','test'):
        if len(split[name]) != len(set(split[name])): raise ValueError('Duplicate split membership')
    if check_counts:
        expected_cameras = {'1','2'} if dataset == 'upfall' else {'0'}
        if {str(r['camera']) for r in records} != expected_cameras:
            raise ValueError('Manifest camera selection disagrees with Table 1')
        if dataset == 'urfall' and len({r['physical_id'] for r in records}) != len(records):
            raise ValueError('UR Fall requires one Camera 0 stream per complete sequence')
    if len(ids) != len(set(ids)): raise ValueError('Duplicate camera-stream ID')
    sets = [set(split[k]) for k in ('train','val','test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('A camera stream crosses partitions')
    if set.union(*sets) != set(ids): raise ValueError('Splits must cover each manifest stream exactly once')
    maps = [{r['subject_id' if dataset == 'upfall' else 'physical_id'] for r in records if r['sequence_id'] in s} for s in sets]
    if any(maps[i] & maps[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Subject or complete physical sequence leakage')
    if check_counts:
        expected = [11,3,3] if dataset == 'upfall' else [49,10,11]
        if [len(s) for s in maps] != expected: raise ValueError('Partition sizes disagree with Table 1')
        if dataset == 'urfall':
            for name,fall,adl in [('train',21,28),('val',4,6),('test',5,6)]:
                c = Counter(bool(r['fall']) for r in records if r['sequence_id'] in split[name])
                if c[True] != fall or c[False] != adl: raise ValueError('UR Fall strata disagree with Table 1')

def generate_split(records, dataset, seed):
    rng = np.random.default_rng(seed)
    out = dict(train=[],val=[],test=[], provenance='generated_partition', split_seed=int(seed))
    if dataset == 'upfall':
        units = sorted({r['subject_id'] for r in records})
        if len(units) != 17: raise ValueError('UP-Fall requires 17 subjects for this protocol')
        units = list(rng.permutation(units))
        groups = [units[:11],units[11:14],units[14:]]
        for name, group in zip(('train','val','test'),groups):
            out[name] = [r['sequence_id'] for r in records if r['subject_id'] in group]
    else:
        for fall,counts in [(True,(21,4,5)),(False,(28,6,6))]:
            units = sorted(r['sequence_id'] for r in records if bool(r['fall']) == fall)
            if len(units) != sum(counts): raise ValueError('Incomplete UR Fall dataset')
            units = list(rng.permutation(units)); start=0
            for name,count in zip(('train','val','test'),counts):
                out[name].extend(units[start:start+count]); start += count
    validate_splits(records,out,dataset)
    return out
