"""Manifest, annotation, array, and clock validation."""
from pathlib import Path
import numpy as np


def validate_records(records, root=None, image_size=None):
    if not records: raise ValueError('Empty manifest')
    seen=set(); physical={};views=set()
    for r in records:
        sid=r['sequence_id']
        if not isinstance(sid,str) or not sid or sid in seen: raise ValueError('Duplicate or invalid sequence ID')
        seen.add(sid)
        view=(r['physical_id'],str(r['camera']))
        if view in views:raise ValueError('Duplicate physical sequence/camera mapping')
        views.add(view)
        n=r['frames']; fps=r['fps']
        if isinstance(n,bool) or not isinstance(n,int) or n<1 or not np.isfinite(fps) or fps<=0:
            raise ValueError('Invalid frame count or FPS')
        if not isinstance(r['fall'],bool): raise ValueError('fall must be boolean')
        for field in ('physical_id','subject_id','camera','activity'):
            if not str(r[field]): raise ValueError('Empty identity field')
        times=[r.get(k) for k in ('onset_seconds','impact_seconds','completion_seconds')]
        if r['fall'] and times[-1] is None: raise ValueError('Fall requires completion time')
        if r['fall'] and times[-1]<=0:raise ValueError('Completion must follow stream start')
        if not r['fall'] and any(t is not None for t in times): raise ValueError('ADL has event times')
        known=[t for t in times if t is not None]
        if any(not np.isfinite(t) or t<0 or t>(n-1)/fps+1e-8 or abs(t*fps-round(t*fps))>1e-6 for t in known):
            raise ValueError('Event time outside frame grid')
        if known != sorted(known): raise ValueError('Onset/impact/completion order is invalid')
        previous=0.
        for a,b in r.get('excluded_intervals',[]):
            if not np.isfinite([a,b]).all() or a<previous or a<0 or b<=a or b>n/fps+1e-8:
                raise ValueError('Exclusions must be ordered disjoint half-open intervals within stream')
            previous=b
        key=r['physical_id']; signature=(r['subject_id'],r['fall'],r.get('completion_seconds'),r.get('impact_seconds'))
        if key in physical and physical[key]!=signature: raise ValueError('Physical-event mapping or synchronized boundaries disagree')
        physical[key]=signature
        if root is not None:
            root=Path(root)
            for field in ('rgb_path','skeleton_path'):
                path=Path(r[field])
                if path.is_absolute() or '..' in path.parts: raise ValueError('Array paths must be manifest-relative')
            rgb=np.load(root/r['rgb_path'],mmap_mode='r',allow_pickle=False)
            pose=np.load(root/r['skeleton_path'],mmap_mode='r',allow_pickle=False)
            if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[0]!=n or rgb.shape[-1]!=3: raise ValueError('Invalid RGB array')
            if image_size and rgb.shape[1:3]!=(image_size,image_size): raise ValueError('RGB size differs from configuration')
            if pose.shape!=(n,17,3) or not np.isfinite(pose).all(): raise ValueError('Invalid skeleton array')
            if ((pose[...,2]<0)|(pose[...,2]>1)).any(): raise ValueError('Pose confidence outside [0,1]')
    return dict(streams=len(records),physical_sequences=len(physical),subjects=len({r['subject_id'] for r in records}))


def valid_frames(record,cfg):
    d=cfg['data'];fps=d['fps'];wo=d['observation_frames'];n=record['frames']
    last=min(n-1,int(round(record['completion_seconds']*fps))-1) if record['fall'] else n-1-int(round(d['bin_edges'][-1]*fps))
    return [f for f in range(wo-1,last+1) if not any(a<=u/fps<b for u in range(f-wo+1,f+1) for a,b in record.get('excluded_intervals',[]))]
