import os
from pathlib import Path
import torch
from pft_et.utils.io import data_fingerprints,array_fingerprints

def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg, threshold, best, selection=None):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, config=cfg,
                    threshold=threshold, threshold_selection=selection, best=best, data_fingerprints=data_fingerprints(cfg),
                    array_fingerprints=array_fingerprints(cfg),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), tmp)
    os.replace(tmp,path)

def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=True)
    if model is not None: model.load_state_dict(ck['model'],strict=True)
    if optimizer is not None: optimizer.load_state_dict(ck['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(ck['scheduler'])
    return ck
