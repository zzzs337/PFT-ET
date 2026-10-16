import torch
from torch import nn
from pft_et.models.transition import transition_samples

class FutureConditionedEventTime(nn.Module):
    def __init__(self,dim,hidden,states,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,states))
    def forward(self,z,dist,samples,mode='sampled'):
        weights=dist['pi']
        if mode=='top1':
            index=weights.argmax(-1);b=torch.arange(z.shape[0],device=z.device)
            dist={key:value[b,index][:,None] for key,value in dist.items()}
            weights=torch.ones_like(dist['pi'])
        r=dist['mu'][:,:,None] if mode=='mean' else transition_samples(dist,samples)
        expanded=z[:,None,None].expand_as(r)
        # Eq. (12) uses [z || r] in the supplied manuscript.
        conditionals=self.net(torch.cat([expanded,r],-1)).softmax(-1)
        component=conditionals.mean(2)  # Eq. (13)
        probability=(weights[...,None]*component).sum(1)  # Eq. (14)
        return probability,component
