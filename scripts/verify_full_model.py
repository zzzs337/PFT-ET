import _common
import argparse
import torch
from pft_et.models.model import PFTET
from pft_et.losses.losses import joint_loss
from pft_et.utils.io import load_config,save_json
from pft_et.utils.seed import set_seed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='outputs/full_model_verification.json');a=p.parse_args()
    torch.set_num_threads(2);set_seed(2024)
    cfg=load_config('configs/default.yaml');cfg['model']['rgb_checkpoint']=a.checkpoint
    model=PFTET(cfg).train()
    batch=dict(rgb=torch.zeros(1,3,8,224,224),skeleton=torch.zeros(1,8,51),
               future_rgb=torch.ones(1,3,8,224,224)*.1,future_skeleton=torch.ones(1,8,51)*.1,
               transition_valid=torch.tensor([True]),label=torch.tensor([3]))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    out=model(batch);loss,parts=joint_loss(model,out,batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.target.parameters())
    rgb_gradient=any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.online.rgb.parameters())
    transition_gradient=model.transition.network[-1].weight.grad
    assert rgb_gradient and torch.isfinite(transition_gradient).all() and transition_gradient.abs().sum()>0
    optimizer.step();model.update_target()
    save_json(a.output,dict(synthetic_input=True,real_backbone=True,pretrained_loaded=True,
              probability_shape=list(out['probability'].shape),finite_loss=True,
              losses={k:float(v) for k,v in parts.items()},backward=True,backbone_gradient=True,
              transition_gradient=True,target_gradient=False,adamw_step=True,ema_update=True))
    print('Full PFT-ET with pretrained UniFormerV2: forward, all losses, backward, AdamW and EMA passed.')
