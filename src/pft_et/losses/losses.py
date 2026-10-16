import torch
import torch.nn.functional as F
from pft_et.models.transition import mixture_log_prob

def event_nll(probability,label):
    return -probability.clamp_min(1e-12).log().gather(1,label[:,None]).mean()

def joint_loss(model,output,batch):
    # Eqs. (10), (16), (17). No additional regularization losses.
    cfg=model.cfg; strategy=cfg['model']['strategy']; label=batch['label']
    evt=event_nll(output['probability'],label)
    aux=event_nll(output['auxiliary'],label) if cfg['model']['use_auxiliary'] and strategy not in ('current','history','distillation') else evt*0
    tr=evt*0
    if cfg['loss']['transition']>0 and strategy not in ('current','history'):
        mask=batch['transition_valid'].bool()
        if mask.any():
            target=model.transition_target(batch)
            if output['regression'] is not None:
                tr=F.mse_loss(output['regression'][mask],target[mask])
            else:
                dist={k:v[mask] for k,v in output['distribution'].items()}
                tr=-mixture_log_prob(target[mask],dist).mean()
    total=evt+cfg['loss']['transition']*tr+cfg['loss']['auxiliary']*aux
    return total,dict(event=evt.detach(),transition=tr.detach(),auxiliary=aux.detach(),total=total.detach())
