import torch
from pft_et.models.transition import mixture_log_prob
from pft_et.metrics.components import component_statistics

def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

@torch.no_grad()
def predict(model,loader,device,analyze=False):
    model.eval();rows=[]
    for batch in loader:
        batch=to_device(batch,device);out=model(batch);p=out['probability']
        analysis={}
        if analyze and 'transition_valid' in batch and (out['distribution'] is not None or out['regression'] is not None):
            target=model.transition_target(batch);dist=out['distribution']
            gaussian=out['regression'] is None and dist is not None
            mean=(dist['pi'][...,None]*dist['mu']).sum(1) if gaussian else out['regression']
            err=(mean-target).square().mean(-1)
            lp=mixture_log_prob(target,dist) if gaussian else None
            for i,valid in enumerate(batch['transition_valid']):
                if valid:
                    r=dict(squared_error=float(err[i]),nll=float(-lp[i]) if lp is not None else None)
                    if gaussian:
                        pi=dist['pi'][i]
                        # Eq. (20)
                        r.update(effective_components=float(torch.exp(-(pi*(pi+1e-12).log()).sum())),maximum_weight=float(pi.max()),
                                 mixture_weights=pi.tolist(),component_probability=out['component_probability'][i].tolist())
                        r.update(component_statistics(pi.cpu(),dist['mu'][i].cpu(),dist['sigma'][i].cpu(),out['component_probability'][i].cpu()))
                    r['coordinate_system']='model_native_ema; not the shared-reference Table 6(a) experiment'
                    analysis[i]=r
        for i in range(p.shape[0]):
            rows.append(dict(sequence_id=batch['sequence_id'][i],time=float(batch['time'][i]),frame=int(batch['frame'][i]),
                             probability=p[i].cpu().tolist(),label=int(batch['label'][i]),tau=float(batch['tau'][i]),transition=analysis.get(i)))
    return rows
