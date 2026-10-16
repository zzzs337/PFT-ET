import torch
import pytest
from pft_et.models.model import PFTET
from pft_et.models.transition import mixture_log_prob
from pft_et.data.dataset import PrefixDataset,make_loader
from pft_et.losses.losses import joint_loss
from pft_et.utils.checkpoint import save_checkpoint,load_checkpoint

def test_mixture_density():
    target=torch.tensor([[0.,0.]])
    dist=dict(pi=torch.tensor([[.25,.75]]),mu=torch.zeros(1,2,2),sigma=torch.ones(1,2,2))
    expected=torch.distributions.MultivariateNormal(torch.zeros(2),torch.eye(2)).log_prob(target[0])
    torch.testing.assert_close(mixture_log_prob(target,dist)[0],expected)

@pytest.mark.parametrize('strategy',['full','current','history','distillation','deterministic'])
def test_forward_and_backward(cfg,strategy):
    cfg['model']['strategy']=strategy
    model=PFTET(cfg);batch=next(iter(make_loader(PrefixDataset(cfg,'train',True),cfg)))
    model.train();out=model(batch)
    torch.testing.assert_close(out['probability'].sum(-1),torch.ones(len(batch['label'])))
    loss,parts=joint_loss(model,out,batch);assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.online.parameters())
    assert all(p.grad is None for p in model.target.parameters())
    if strategy=='full':
        assert model.transition.network[-1].weight.grad.abs().sum()>0
    model.eval();out=model({k:v for k,v in batch.items() if not k.startswith('future') and k!='transition_valid'})
    assert out['probability'].shape==(4,7)

@pytest.mark.parametrize('mode',['sampled','mean','top1'])
def test_marginalization(cfg,mode):
    cfg['model']['marginalization']=mode;model=PFTET(cfg).eval()
    batch=next(iter(make_loader(PrefixDataset(cfg,'train'),cfg)));out=model(batch)
    torch.testing.assert_close(out['probability'].sum(-1),torch.ones(4))
    if mode!='top1':
        torch.testing.assert_close(out['probability'],(out['distribution']['pi'][...,None]*out['component_probability']).sum(1))

def test_ema_mask_and_checkpoint(cfg,tmp_path):
    model=PFTET(cfg);old=next(model.target.parameters()).clone()
    with torch.no_grad(): next(model.online.parameters()).add_(1)
    model.update_target();torch.testing.assert_close(next(model.target.parameters()),old+.01)
    batch=next(iter(make_loader(PrefixDataset(cfg,'train',True),cfg)))
    batch['transition_valid'].fill_(False)
    out=model(batch);loss,parts=joint_loss(model,out,batch);assert parts['transition']==0
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1)
    path=tmp_path/'checkpoint.pt';save_checkpoint(path,model,optimizer,scheduler,0,cfg,.5,[.9,0,0])
    clone=PFTET(cfg);load_checkpoint(path,clone)
    for a,b in zip(model.parameters(),clone.parameters()): torch.testing.assert_close(a,b)


def test_target_is_future_minus_current_in_ema_space(cfg):
    from torch import nn
    class KnownEncoder(nn.Module):
        def forward(self,rgb,skeleton): return skeleton.mean(1)[:,:16]
    model=PFTET(cfg);model.target=KnownEncoder()
    batch=next(iter(make_loader(PrefixDataset(cfg,'train',True),cfg)))
    batch['skeleton'].fill_(1);batch['future_skeleton'].fill_(3)
    target=model.transition_target(batch)
    torch.testing.assert_close(target,torch.full((4,16),2.))

def test_default_research_encoder_uses_real_backbone(cfg):
    from pft_et.utils.io import load_config
    c=load_config('configs/default.yaml')
    assert c['model']['rgb_backend']=='uniformerv2' and not c['experiment']['synthetic']

def test_fusion_difference_matches_absolute_difference_in_eq5(cfg):
    model=PFTET(cfg).eval()
    batch=next(iter(make_loader(PrefixDataset(cfg,'train'),cfg)))
    seen=[]
    handle=model.online.gate.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach()))
    model(batch);handle.remove()
    features=seen[0];d=cfg['model']['dim']
    torch.testing.assert_close(features[:,2*d:],(features[:,:d]-features[:,d:2*d]).abs())
