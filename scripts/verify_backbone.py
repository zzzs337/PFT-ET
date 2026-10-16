import _common
import argparse
import hashlib
from pathlib import Path
import torch
from pft_et.models.encoder import UniFormerV2RGB
from pft_et.utils.io import save_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',default='outputs/backbone_verification.json');a=p.parse_args()
    torch.set_num_threads(2)
    model=UniFormerV2RGB(a.checkpoint,initialize=a.checkpoint is not None).eval()
    with torch.no_grad(): feature=model(torch.zeros(1,3,8,224,224))
    assert feature.shape==(1,768) and torch.isfinite(feature).all()
    digest=None
    if a.checkpoint:
        h=hashlib.sha256()
        with Path(a.checkpoint).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        digest=h.hexdigest()
    save_json(a.output,dict(parameters=sum(p.numel() for p in model.parameters()),input_shape=[1,3,8,224,224],
         output_shape=list(feature.shape),finite=True,pretrained_loaded=a.checkpoint is not None,sha256=digest,torch_version=torch.__version__))
