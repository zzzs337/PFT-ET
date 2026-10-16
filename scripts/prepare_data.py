import _common
import argparse
import csv
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
from pft_et.data.preprocessing import resampling_indices,normalize_skeleton,resize_rgb
from pft_et.utils.io import read_jsonl,write_jsonl
from pft_et.data.integrity import validate_records

def read_source(record,root):
    timestamps=np.loadtxt(root/record['timestamps'],delimiter=',',ndmin=1)
    if timestamps.ndim!=1: raise ValueError('timestamps CSV: one timestamp in seconds per row, no header')
    if 'frame_list' in record:
        paths=(root/record['frame_list']).read_text().splitlines()
        frames=[np.asarray(Image.open(root/p).convert('RGB')) for p in paths]
    else:
        cap=cv2.VideoCapture(str(root/record['video']))
        frames=[]
        while True:
            ok,bgr=cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        cap.release()
    if len(frames)!=len(timestamps): raise ValueError('Frame count disagrees with original timestamp count')
    if not frames: raise ValueError('No RGB frames')
    return frames,timestamps

def prepare(manifest,out,fps=20,size=224,overwrite=False):
    root=Path(manifest).resolve().parent;out=Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite: raise FileExistsError('Processed output is nonempty')
    out.mkdir(parents=True,exist_ok=True);rows=[];seen=set()
    for record in read_jsonl(manifest):
        sid=record['sequence_id']
        if sid in seen or '/' in sid or '..' in sid or '\\' in sid: raise ValueError('Invalid or duplicate stream ID')
        seen.add(sid)
        frames,timestamps=read_source(record,root);height,width=frames[0].shape[:2]
        if any(frame.shape != frames[0].shape for frame in frames): raise ValueError('Source image dimensions must be constant')
        grid,idx=resampling_indices(timestamps,fps)
        poses=np.load(root/record['skeleton'],allow_pickle=False)
        if poses.shape!=(len(frames),17,3): raise ValueError('Raw pose must be [source_frames,17,3]')
        pose=normalize_skeleton(poses,width,height)[idx]
        target=out/sid;target.mkdir(parents=True,exist_ok=True)
        rgb=np.lib.format.open_memmap(target/'rgb.npy',mode='w+',dtype=np.uint8,shape=(len(grid),size,size,3))
        for j,source in enumerate(idx): rgb[j]=resize_rgb(frames[source],size)
        rgb.flush();del rgb
        np.save(target/'skeleton.npy',pose);np.save(target/'timestamps.npy',grid)
        meta={k:record[k] for k in ['sequence_id','physical_id','subject_id','camera','activity','fall']}
        for key in ('event_id','annotation_source','annotation_version','impact_annotation_source'):
            if key in record: meta[key]=record[key]
        meta.update(fps=fps,frames=len(grid),rgb_path=f'{sid}/rgb.npy',skeleton_path=f'{sid}/skeleton.npy',
                    source_start_seconds=float(timestamps[0]),excluded_intervals=record.get('excluded_intervals',[]))
        # Annotation seconds are relative to the first source frame; nearest 20-FPS boundary.
        for key in ['onset_seconds','impact_seconds','completion_seconds']:
            value=record.get(key)
            meta[key]=float(np.rint(value*fps)/fps) if value is not None else None
        if record['fall'] and (meta['completion_seconds'] is None or not 0<meta['completion_seconds']<=grid[-1]):
            raise ValueError('A fall requires an evaluable completion annotation')
        if meta['impact_seconds'] is not None and not 0<=meta['impact_seconds']<=meta['completion_seconds']:
            raise ValueError('First impact must not follow completion')
        rows.append(meta)
    # Synchronized views use a common physical event reference after clock alignment.
    groups={}
    for r in rows: groups.setdefault(r['physical_id'],[]).append(r)
    for group in groups.values():
        for key in ('completion_seconds','impact_seconds'):
            values={r[key] for r in group if r['fall']}
            if len(values)>1: raise ValueError(f'Synchronized streams disagree on {key}; align clocks before preparation')
    validate_records(rows,out,size)
    write_jsonl(out/'manifest.jsonl',rows)
    return out/'manifest.jsonl'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fps',type=int,default=20);p.add_argument('--image-size',type=int,default=224);p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();print(prepare(a.manifest,a.output,a.fps,a.image_size,a.overwrite))
