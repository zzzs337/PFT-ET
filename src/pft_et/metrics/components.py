"""Pairwise Bhattacharyya overlap, mean separation, and weighted JS divergence."""
import numpy as np


def component_statistics(pi,mu,sigma,prob):
    pi,mu,sigma,prob=map(lambda x:np.asarray(x,dtype=float),(pi,mu,sigma,prob))
    if sigma.shape!=mu.shape or len(pi)!=len(mu) or (sigma<=0).any(): raise ValueError('Invalid component inputs')
    overlap=[];separation=[];weights=[]
    for i in range(len(pi)):
        for j in range(i+1,len(pi)):
            avg=(sigma[i]**2+sigma[j]**2)/2; delta=mu[i]-mu[j]
            db=(delta**2/avg).sum()/8 + .5*np.log(avg/(sigma[i]*sigma[j])).sum()
            overlap.append(float(np.exp(-db)));separation.append(float(np.sqrt((delta**2/avg).sum())))
            weights.append(float(pi[i]*pi[j]))
    mixed=(pi[:,None]*prob).sum(0);js=0.
    for weight,p in zip(pi,prob):
        mid=(p+mixed)/2
        def kl(a,b):return np.sum(np.where(a>0,a*np.log(np.maximum(a,1e-300)/np.maximum(b,1e-300)),0))
        js+=weight*.5*(kl(p,mid)+kl(mixed,mid))
    return dict(component_overlap=float(np.average(overlap,weights=weights)) if sum(weights)>0 else None,
                component_separation=float(np.average(separation,weights=weights)) if sum(weights)>0 else None,
                event_head_js=float(js),pair_weighting='pi_i*pi_j normalized over unordered pairs',
                separation_dimension_normalization='none')
