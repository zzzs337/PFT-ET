import math
import torch
from torch import nn
import torch.nn.functional as F

class ConditionalFutureTransition(nn.Module):
    def __init__(self,dim,hidden,mixtures,sigma_min,dropout):
        super().__init__(); self.k=mixtures;self.d=dim;self.sigma_min=sigma_min
        self.network=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,mixtures*(2*dim+1)))
    def forward(self,z):
        raw=self.network(z).reshape(-1,self.k,2*self.d+1)
        # Eq. (9)
        return dict(pi=raw[...,0].softmax(-1),mu=raw[...,1:self.d+1],
                    sigma=F.softplus(raw[...,self.d+1:])+self.sigma_min)

def mixture_log_prob(target, dist):
    # Eqs. (9)-(10): diagonal dimensions sum; modes use stable log-sum-exp.
    target=target.float();mu=dist['mu'].float();sigma=dist['sigma'].float()
    logp=(-0.5*((target[:,None]-mu)/sigma).square()-sigma.log()-0.5*math.log(2*math.pi)).sum(-1)
    return torch.logsumexp(dist['pi'].float().clamp_min(1e-12).log()+logp,dim=-1)

def transition_samples(dist, samples):
    # Eq. (11), gradients flow to both means and scales.
    mu,sigma=dist['mu'],dist['sigma']
    eps=torch.randn(mu.shape[0],mu.shape[1],samples,mu.shape[2],device=mu.device,dtype=mu.dtype)
    return mu[:,:,None]+sigma[:,:,None]*eps
