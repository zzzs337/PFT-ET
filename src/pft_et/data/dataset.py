import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pft_et.data.preprocessing import uniform_indices
from pft_et.data.splits import validate_splits
from pft_et.utils.io import read_jsonl
from pft_et.utils.seed import worker_seed
from pft_et.data.integrity import validate_records

def label_at(frame, completion_frame, fall, edges_frames):
    # Eqs. (1)-(4); zero-based Python states, last state is S_H.
    if not fall: return len(edges_frames)-2+1
    tau = completion_frame-frame
    if tau <= 0: raise ValueError('Fall labels require t<T_F')
    return int(np.searchsorted(edges_frames[1:],tau,side='left'))

def excluded(record, time):
    return any(a <= time < b for a,b in record.get('excluded_intervals',[]))

class PrefixDataset(Dataset):
    def __init__(self, cfg, partition, include_future=False, all_positions=False):
        self.cfg, self.include_future = cfg, include_future
        self.records = read_jsonl(cfg['data']['manifest'])
        self.root = Path(cfg['data']['manifest']).resolve().parent
        validate_records(self.records)
        split = json.loads(Path(cfg['data']['splits']).read_text())
        validate_splits(self.records,split,cfg['data']['dataset'],not cfg['experiment']['synthetic'])
        self.records = [r for r in self.records if r['sequence_id'] in split[partition]]
        d = cfg['data']; fps=d['fps']; wo=d['observation_frames']
        self.edges = np.rint(np.array(d['bin_edges'])*fps).astype(int)
        self.hf = int(round(d['future_seconds']*fps)); self.index=[]
        for ri,r in enumerate(self.records):
            if abs(r['fps']-fps)>1e-9: raise ValueError('Prepared FPS disagrees with config')
            n=r['frames']; tf=int(round(r['completion_seconds']*fps)) if r['fall'] else -1
            last = n-1 if all_positions else (min(n-1,tf-1) if r['fall'] else n-1-self.edges[-1])
            for t in range(wo-1,last+1):
                # Exclusion anywhere in the current window invalidates this prefix.
                if any(excluded(r,u/fps) for u in range(t-wo+1,t+1)): continue
                self.index.append((ri,t))
        if not self.index: raise ValueError(f'No valid prefixes in {partition}')

    @lru_cache(maxsize=2)
    def arrays(self, ri):
        r=self.records[ri]
        if not hasattr(self,'_validated_arrays'):self._validated_arrays=set()
        if ri not in self._validated_arrays:
            validate_records([r], self.root, self.cfg['data']['image_size']);self._validated_arrays.add(ri)
        return (np.load(self.root/r['rgb_path'],mmap_mode='r',allow_pickle=False),
                np.load(self.root/r['skeleton_path'],mmap_mode='r',allow_pickle=False))

    def window(self, ri, start, end):
        d=self.cfg['data']; rgb,ske=self.arrays(ri)
        idx = uniform_indices(end-start+1,d['sampled_frames'])+start
        x=torch.from_numpy(np.asarray(rgb[idx]).copy()).float().permute(3,0,1,2)/255
        mean=torch.tensor(d['rgb_mean'])[:,None,None,None]
        std=torch.tensor(d['rgb_std'])[:,None,None,None]
        return (x-mean)/std,torch.from_numpy(np.asarray(ske[idx]).copy()).float().flatten(1)

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        ri,t=self.index[i]; r=self.records[ri]; d=self.cfg['data']; fps=d['fps']
        rgb,ske=self.window(ri,t-d['observation_frames']+1,t)
        tf=int(round(r['completion_seconds']*fps)) if r['fall'] else -1
        valid_label = (t < tf) if r['fall'] else (t+self.edges[-1] <= r['frames']-1)
        y=label_at(t,tf,r['fall'],self.edges) if valid_label else -1
        item=dict(rgb=rgb,skeleton=ske,label=torch.tensor(y),frame=t,
                  sequence_id=r['sequence_id'],time=t/fps,
                  tau=(tf-t)/fps if r['fall'] else -1.0)
        if self.include_future:
            valid = t+self.hf <= r['frames']-1 and not any(excluded(r,u/fps) for u in range(t+1,t+self.hf+1))
            if valid: frgb,fske=self.window(ri,t+1,t+self.hf)
            else: frgb,fske=torch.zeros_like(rgb),torch.zeros_like(ske)
            item.update(future_rgb=frgb,future_skeleton=fske,transition_valid=torch.tensor(valid))
        if self.cfg['model']['strategy'] == 'history':
            hrs=[]; hss=[]
            for k in range(self.cfg['model']['history_windows'],0,-1):
                end=max(d['observation_frames']-1,t-k*self.cfg['model']['history_stride_frames'])
                a,b=self.window(ri,end-d['observation_frames']+1,end); hrs.append(a);hss.append(b)
            item.update(history_rgb=torch.stack(hrs),history_skeleton=torch.stack(hss))
        return item

def collate_fn(items):
    return {k: torch.stack([x[k] for x in items]) if isinstance(items[0][k],torch.Tensor)
            else torch.tensor([x[k] for x in items],dtype=torch.float64 if isinstance(items[0][k],float) else torch.long) if isinstance(items[0][k],(int,float))
            else [x[k] for x in items] for k in items[0]}

def make_loader(dataset, cfg, training=False):
    g=torch.Generator().manual_seed(cfg['training']['seed'])
    return DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=training,
                      num_workers=cfg['data']['workers'],collate_fn=collate_fn,
                      worker_init_fn=worker_seed,generator=g,pin_memory=torch.cuda.is_available())
