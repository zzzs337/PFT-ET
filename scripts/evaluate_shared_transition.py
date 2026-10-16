"""Evaluate model exports in an explicitly common, frozen transition coordinate system.

Each JSONL row: {"sequence_id":str,"frame":int,"target":[d floats]}
Prediction: same identifiers and either {"mean":[d]} or
{"pi":[k],"mu":[[d],...],"sigma":[[d],...]}.
All values must already be in the same standardized frozen-reference coordinates.
This is an evaluation utility: it does not construct or train the frozen encoder.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, ndtr


def load_rows(filename):
    result={}
    for line in Path(filename).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line); key=(str(row['sequence_id']),int(row['frame']))
        if key in result: raise ValueError(f'duplicate row {key}')
        result[key]=row
    if not result:raise ValueError('empty file: '+str(filename))
    return result


def quantiles(pi,mu,sigma,qs):
    # Coordinate-wise marginal quantiles for a mixture of diagonal Gaussians.
    lo=(mu-12*sigma).min(axis=0);hi=(mu+12*sigma).max(axis=0)
    output=[]
    for q in qs:
        l=lo.copy();h=hi.copy()
        for _ in range(64):
            mid=(l+h)/2
            cdf=(pi[:,None]*ndtr((mid[None,:]-mu)/sigma)).sum(axis=0)
            l=np.where(cdf<q,mid,l);h=np.where(cdf>=q,mid,h)
        output.append((l+h)/2)
    return output


def calculate(targets,predictions):
    if not targets:raise ValueError('No transition targets')
    if targets.keys()!=predictions.keys():
        raise ValueError('target and prediction IDs/frames differ; no implicit matching')
    errors=[];nll=[];cover={q:[] for q in (.5,.8,.9)};width={q:[] for q in (.5,.8,.9)};kind=None;dim=None
    for key in sorted(targets):
        y=np.asarray(targets[key]['target'],float)
        if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError(f'invalid target {key}')
        if dim is None:dim=len(y)
        if len(y)!=dim:raise ValueError('inconsistent target dimensions')
        p=predictions[key]
        if 'mean' in p:
            if kind not in (None,'deterministic'):raise ValueError('mixed prediction families')
            kind='deterministic';estimate=np.asarray(p['mean'],float)
            if estimate.shape!=y.shape or not np.isfinite(estimate).all():raise ValueError('invalid prediction mean')
        else:
            if kind not in (None,'mixture'):raise ValueError('mixed prediction families')
            kind='mixture';pi=np.asarray(p['pi'],float);mu=np.asarray(p['mu'],float);sigma=np.asarray(p['sigma'],float)
            if pi.ndim!=1 or mu.shape!=(len(pi),dim) or sigma.shape!=mu.shape:raise ValueError('invalid mixture shapes')
            if not np.isfinite(pi).all() or not np.isfinite(mu).all() or not np.isfinite(sigma).all():raise ValueError('nonfinite mixture')
            if (pi<0).any() or abs(pi.sum()-1)>1e-6 or (sigma<=0).any():raise ValueError('invalid mixture weights/scales')
            estimate=np.sum(pi[:,None]*mu,axis=0)
            lp=(-.5*((y[None,:]-mu)/sigma)**2-np.log(sigma)-.5*math.log(2*math.pi)).sum(axis=1)
            nll.append(float(-logsumexp(lp,b=pi)))
            quantile_levels=sorted(set(v for q in cover for v in ((1-q)/2,(1+q)/2)))
            vals=dict(zip(quantile_levels,quantiles(pi,mu,sigma,quantile_levels)))
            for q in cover:
                low=vals[(1-q)/2];upper=vals[(1+q)/2]
                cover[q].append(float(((y>=low)&(y<=upper)).mean()))
                width[q].append(float((upper-low).mean()))
        errors.append(float(np.mean((y-estimate)**2)))
    result=dict(family=kind,dimension=dim,n_prefixes=len(errors),transition_rmse=float(np.sqrt(np.mean(errors))),
         transition_nll=float(np.mean(nll)) if nll else None,
         representation='provided shared frozen-reference standardized coordinates',
         scope='common-coordinate scores; reference export and predictor provenance are stored by reference_transition.py')
    if nll:
        result['coordinate_coverage']={str(q):dict(empirical=float(np.mean(cover[q])),mean_width=float(np.mean(width[q]))) for q in cover}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--targets',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();summary=calculate(load_rows(a.targets),load_rows(a.predictions))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
