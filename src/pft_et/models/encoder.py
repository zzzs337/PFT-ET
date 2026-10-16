import math
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn

@lru_cache(maxsize=1)
def upstream_uniformerv2():
    import mmaction
    # Load the installed upstream backbone without importing unrelated localizers.
    # MMAction2's 1.2.0 PyPI wheel omits DRN files imported by models/__init__.py.
    path = Path(mmaction.__file__).parent / 'models/backbones/uniformerv2.py'
    name = 'pft_et_upstream_uniformerv2'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.UniFormerV2

class UniFormerV2RGB(nn.Module):
    """Official MMAction2 B/16 backbone, no classification head, [B,3,8,224,224]."""
    def __init__(self, checkpoint, initialize=True, dropout=0.3):
        super().__init__()
        self.backbone = upstream_uniformerv2()(input_resolution=224,
            patch_size=16,width=768,layers=12,heads=12,t_size=8,dw_reduction=1.5,
            backbone_drop_path_rate=0.,temporal_downsample=False,no_lmhra=True,
            double_lmhra=True,return_list=[8,9,10,11],n_layers=4,n_dim=768,n_head=12,
            mlp_factor=4.,drop_path_rate=0.,mlp_dropout=[dropout]*4,clip_pretrained=False)
        if initialize:
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError('Supply model.rgb_checkpoint with the Kinetics-710 B/16 8-frame weights; see pretrained/README.md')
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)
            state = state.get('state_dict',state.get('model',state))
            cleaned={}
            for k,v in state.items():
                k=k.removeprefix('module.')
                if k.startswith('backbone.'): cleaned[k[len('backbone.'):]]=v
            if not cleaned:
                cleaned={k:v for k,v in state.items() if k in self.backbone.state_dict()}
            # Strict loading prevents an unnoticed random backbone or wrong resource.
            self.backbone.load_state_dict(cleaned,strict=True)

    def forward(self, rgb):
        if rgb.shape[1:] != (3,8,224,224): raise ValueError('UniFormerV2 expects [B,3,8,224,224]')
        x=self.backbone(rgb)
        if not isinstance(x,torch.Tensor) or x.ndim != 2 or x.shape[-1] != 768:
            raise ValueError('Unexpected MMAction2 UniFormerV2 feature shape')
        return x

class SmokeRGB(nn.Module):
    """Small learned RGB encoder used only by the synthetic execution-chain check."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(3,8,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(8,768))
    def forward(self,rgb): return self.net(rgb)

def temporal_encoding(length, dim, device, dtype):
    pos=torch.arange(length,device=device).float()[:,None]
    freq=torch.exp(torch.arange(0,dim,2,device=device).float()*(-math.log(10000)/dim))
    pe=torch.zeros(length,dim,device=device)
    pe[:,0::2]=torch.sin(pos*freq); pe[:,1::2]=torch.cos(pos*freq[:pe[:,1::2].shape[1]])
    return pe.to(dtype)

class MotionEncoder(nn.Module):
    def __init__(self,cfg,initialize=True):
        super().__init__(); m=cfg['model']; d=m['dim']
        self.rgb=UniFormerV2RGB(m['rgb_checkpoint'],initialize,m['dropout']) if m['rgb_backend']=='uniformerv2' else SmokeRGB()
        self.rgb_projection=nn.Linear(768,d)
        self.skeleton_projection=nn.Linear(51,d)
        layer=nn.TransformerEncoderLayer(d,m['skeleton_heads'],m['skeleton_ffn'],m['dropout'],batch_first=True)
        self.skeleton_encoder=nn.TransformerEncoder(layer,m['skeleton_layers'],enable_nested_tensor=False)
        self.gate=nn.Linear(3*d,1)
        self.state=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(m['dropout']))
    def forward(self,rgb,skeleton):
        if skeleton.ndim != 3 or skeleton.shape[-1]!=51: raise ValueError('Skeleton input must be [B,T,51]')
        fr=self.rgb_projection(self.rgb(rgb))
        x=self.skeleton_projection(skeleton)
        fs=self.skeleton_encoder(x+temporal_encoding(x.shape[1],x.shape[2],x.device,x.dtype)).mean(1)
        # Eq. (5): scalar fusion gate, absolute cross-modal difference.
        alpha=torch.sigmoid(self.gate(torch.cat([fr,fs,(fr-fs).abs()],-1)))
        return self.state(alpha*fr+(1-alpha)*fs)
