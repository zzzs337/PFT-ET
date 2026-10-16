import copy
import json
import os
from functools import lru_cache
from pathlib import Path
import yaml

def merge(base, updates):
    result = copy.deepcopy(base)
    for k, v in updates.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else copy.deepcopy(v)
    return result

def load_config(path):
    path = Path(path)
    cfg = json.loads(path.read_text()) if path.suffix.lower()=='.json' else yaml.safe_load(path.read_text())
    parent = cfg.pop('extends', None)
    cfg = merge(load_config(path.parent / parent), cfg) if parent else cfg
    validate_config(cfg)
    return cfg

def validate_config(c):
    d, m, a = c['data'], c['model'], c['alarm']
    edges = d['bin_edges']
    if edges[0] != 0 or any(b <= a for a,b in zip(edges,edges[1:])):
        raise ValueError('bin_edges must increase strictly from zero')
    if not 0 < d['future_seconds'] <= edges[-1]:
        raise ValueError('future_seconds must lie in (0,H]')
    if a['horizon_seconds'] not in edges[1:]:
        raise ValueError('Alarm horizon must be a bin right boundary')
    if not 1 <= a['k'] <= a['w'] or m['dim'] % m['skeleton_heads']:
        raise ValueError('Invalid confirmation rule or attention dimensions')
    if m['rgb_backend'] not in ('uniformerv2', 'smoke_cnn'):
        raise ValueError('Unknown RGB backend')
    if m['rgb_backend'] == 'smoke_cnn' and not c['experiment'].get('synthetic', False):
        raise ValueError('smoke_cnn is permitted only for synthetic pipeline checks')

def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(tmp,path)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    os.replace(tmp,path)


def file_sha256(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def data_fingerprints(cfg):
    return {key:file_sha256(cfg['data'][key]) for key in ['manifest','splits']}


@lru_cache(maxsize=8192)
def _stat_keyed_hash(path,size,mtime_ns):
    return file_sha256(path)


def array_fingerprints(cfg):
    root=Path(cfg['data']['manifest']).resolve().parent;result={}
    for r in read_jsonl(cfg['data']['manifest']):
        for field in ('rgb_path','skeleton_path'):
            path=(root/r[field]).resolve();stat=path.stat()
            result[r['sequence_id']+'/'+field]=_stat_keyed_hash(str(path),stat.st_size,stat.st_mtime_ns)
    return result
