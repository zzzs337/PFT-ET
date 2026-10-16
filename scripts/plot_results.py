import _common
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pft_et.utils.io import read_jsonl

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summaries',nargs='*',default=[]);p.add_argument('--labels',nargs='*',default=[]);p.add_argument('--predictions');p.add_argument('--sequence');p.add_argument('--frame',type=int);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.summaries:
        labels=a.labels or [Path(x).parent.name for x in a.summaries]
        if len(labels)!=len(a.summaries): p.error('One label per summary')
        summaries=[json.loads(Path(x).read_text()) for x in a.summaries]
        fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        for s,label in zip(summaries,labels):
            curve=s['recall_lead_curve'];x=[r['lead_seconds'] for r in curve]
            for ax,key in zip(axes,['completion_recall','impact_recall']): ax.plot(x,[r[key] for r in curve],label=label)
        for ax,title in zip(axes,['Completion-relative recall','Pre-impact recall']):
            ax.set(xlabel='Minimum lead time (s)',ylabel='Recall',title=title,ylim=(0,1.02));ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'recall_lead.png',dpi=320);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(9,3.8))
        for ax,key,title in zip(axes,['event_f1','event_time_nll'],['Event F1','Event-Time NLL']):
            vals=[next(r['mean'] for r in s['metrics'] if r['metric']==key) for s in summaries]
            ax.bar(labels,vals);ax.set_title(title);ax.tick_params(axis='x',rotation=20)
        fig.tight_layout();fig.savefig(out/'comparison.png',dpi=320);plt.close(fig)
    if a.predictions:
        rows=[r for r in read_jsonl(a.predictions) if a.sequence is None or r['sequence_id']==a.sequence]
        streams=sorted({r['sequence_id'] for r in rows})
        fig,ax=plt.subplots(figsize=(7,3.6))
        for stream in streams:
            seq=[r for r in rows if r['sequence_id']==stream]
            ax.plot([r['time'] for r in seq],[sum(r['probability'][:-1]) for r in seq],label=stream)
        ax.set(xlabel='Time (s)',ylabel='Cumulative fall probability');ax.legend();fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=320);plt.close(fig)
        if a.frame is not None:
            r=next(r for r in rows if r['frame']==a.frame)
            if r['transition'] is None or 'component_probability' not in r['transition']: raise ValueError('Use evaluate --analysis to produce conditional probabilities')
            values=np.asarray(r['transition']['component_probability']);fig,ax=plt.subplots(figsize=(7,3.5))
            im=ax.imshow(values,vmin=0,vmax=1,aspect='auto');fig.colorbar(im,ax=ax,label='Probability')
            ax.set(xlabel='Event-time state (last column: tail)',ylabel='Future component');fig.tight_layout();fig.savefig(out/'conditional_event_time.png',dpi=320);plt.close(fig)
