import copy
import torch
from torch import nn
from pft_et.models.encoder import MotionEncoder
from pft_et.models.transition import ConditionalFutureTransition
from pft_et.models.event_time import FutureConditionedEventTime

class PFTET(nn.Module):
    def __init__(self,cfg,initialize=True):
        super().__init__();self.cfg=cfg; m=cfg['model'];d=m['dim'];h=m['head_hidden'];c=len(cfg['data']['bin_edges'])
        self.online=MotionEncoder(cfg,initialize)
        self.target=copy.deepcopy(self.online)
        self.target.requires_grad_(False);self.target.eval()
        self.transition=ConditionalFutureTransition(d,h,m['mixtures'],m['sigma_min'],m['dropout'])
        self.event_time=FutureConditionedEventTime(d,h,c,m['dropout'])
        self.auxiliary=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(m['dropout']),nn.Linear(h,c))
        self.regression=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(m['dropout']),nn.Linear(h,d))
        self.history=nn.GRU(d,d,batch_first=True) if m['strategy']=='history' else None

    def train(self,mode=True):
        super().train(mode);self.target.eval();return self

    @torch.no_grad()
    def update_target(self):
        # Eq. (7), after each optimizer update.
        beta=self.cfg['model']['ema_beta'] if self.cfg['model']['use_ema'] else 0.
        for target,online in zip(self.target.parameters(),self.online.parameters()):
            target.mul_(beta).add_(online,alpha=1-beta)
        for target,online in zip(self.target.buffers(),self.online.buffers()): target.copy_(online)

    @torch.no_grad()
    def transition_target(self,batch):
        valid=batch['transition_valid'].bool()
        result=torch.zeros(len(valid),self.cfg['model']['dim'],device=batch['rgb'].device)
        if valid.any():
            encoder=self.target if self.cfg['model']['use_ema'] else self.online
            was_training=encoder.training;encoder.eval()
            # Eq. (8): both current and future states come from the same target encoder.
            result[valid]=encoder(batch['future_rgb'][valid],batch['future_skeleton'][valid])-encoder(batch['rgb'][valid],batch['skeleton'][valid])
            encoder.train(was_training)
        return result

    def forward(self,batch,samples=None):
        m=self.cfg['model']; strategy=m['strategy'];z=self.online(batch['rgb'],batch['skeleton'])
        if strategy=='history':
            b,l=batch['history_rgb'].shape[:2]
            hz=self.online(batch['history_rgb'].flatten(0,1),batch['history_skeleton'].flatten(0,1)).reshape(b,l,-1)
            z=self.history(torch.cat([hz,z[:,None]],1))[0][:,-1]
        return self.from_state(z,samples=samples)

    def from_state(self,z,samples=None,distribution=None,intervention=None):
        m=self.cfg['model'];strategy=m['strategy']
        aux=self.auxiliary(z).softmax(-1) if self.training or strategy in ('current','history','distillation') else None
        if strategy in ('current','history','distillation'):
            p=aux;dist=None;component=None
            regression=self.regression(z) if strategy=='distillation' else None
        elif strategy=='deterministic':
            regression=self.regression(z)
            dist=dict(pi=torch.ones(z.shape[0],1,device=z.device),mu=regression[:,None],sigma=torch.zeros_like(regression[:,None]))
            p,component=self.event_time(z,dist,1,'mean')
        else:
            dist=self.transition(z) if distribution is None else distribution;regression=None
            if intervention=='mixture_mean':
                mean=(dist['pi'][...,None]*dist['mu']).sum(1,keepdim=True)
                dist=dict(pi=torch.ones_like(dist['pi'][:,:1]),mu=mean,sigma=torch.zeros_like(mean))
            samples=samples or (m['train_samples'] if self.training else m['test_samples'])
            p,component=self.event_time(z,dist,samples,m['marginalization'])
        return dict(probability=p,auxiliary=aux,distribution=dist,component_probability=component,regression=regression)
