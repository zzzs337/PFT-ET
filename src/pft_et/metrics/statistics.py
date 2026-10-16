import numpy as np
from scipy.stats import chi2

def poisson_interval(count,hours,alpha=0.05):
    if hours<=0: return [None,None]
    low=0. if count==0 else chi2.ppf(alpha/2,2*count)/2/hours
    high=chi2.ppf(1-alpha/2,2*(count+1))/2/hours
    return [float(low),float(high)]

def mean_sd(values):
    x=np.asarray([v for v in values if v is not None],float)
    return dict(mean=float(x.mean()) if len(x) else None,
                sd=float(x.std(ddof=1)) if len(x)>1 else None,n=int(len(x)))

def cluster_bootstrap(run_rows,samples=10000,seed=3407):
    """Sample shared subject/physical-sequence clusters jointly across runs/cameras."""
    from pft_et.metrics.evaluation import summarize_rows
    clusters=sorted({r['cluster_id'] for rows in run_rows for r in rows})
    if not clusters: return {}
    if any({r['cluster_id'] for r in rows} != set(clusters) for rows in run_rows):
        raise ValueError('Cluster bootstrap requires identical evaluated cluster sets across runs')
    grouped=[{c:[r for r in rows if r['cluster_id']==c] for c in clusters} for rows in run_rows]
    rng=np.random.default_rng(seed);values={k:[] for k in ['event_f1','far','alt','recall','preimpact_recall@0.5']}
    for _ in range(samples):
        selected=rng.choice(clusters,len(clusters),replace=True)
        summaries=[summarize_rows([r for c in selected for r in g[c]]) for g in grouped]
        for k in values:
            if k=='far':
                hours=sum(s['adl_hours'] for s in summaries)
                if hours:values[k].append(sum(s['adl_false_alarms'] for s in summaries)/hours)
                continue
            valid=[s[k] for s in summaries if s[k] is not None]
            if valid: values[k].append(float(np.mean(valid)))
    return {k:[float(x) for x in np.percentile(v,[2.5,97.5])] if v else [None,None] for k,v in values.items()}
