from collections import deque
import numpy as np

class AlarmConfirmation:
    """Eq. (19): causal K-of-W; reset at a missing valid prediction position."""
    def __init__(self,k=3,w=5):
        if not 1<=k<=w: raise ValueError('Invalid K-of-W parameters')
        self.k,self.w=k,w;self.history=deque(maxlen=w);self.confirmed=False;self.last=None
    def step(self,risk,threshold,time,stride):
        if self.last is not None and abs(time-self.last-stride)>1e-7:
            self.history.clear();self.confirmed=False
        self.history.append(bool(risk>=threshold))
        active=sum(self.history)>=self.k
        onset=active and not self.confirmed
        self.confirmed=active;self.last=time
        return bool(risk>=threshold),active,onset

def alarm_events(times,risk,threshold,k,w,fps):
    if len(times)!=len(risk) or not np.isfinite(times).all() or not np.isfinite(risk).all(): raise ValueError('Invalid alarm series')
    if fps<=0 or not 0<=threshold<=1 or np.any(np.diff(times)<=0): raise ValueError('Invalid threshold, FPS, or chronology')
    state=AlarmConfirmation(k,w);events=[];candidates=[];confirmed=[]
    for t,p in zip(times,risk):
        c,a,onset=state.step(float(p),threshold,float(t),1/fps)
        candidates.append(c);confirmed.append(a)
        if onset: events.append(float(t))
    return events,np.array(candidates),np.array(confirmed)

def match_events(onsets,falls,horizon):
    """Sec. 4.3 / Appendix A.2: earliest eligible unmatched onset, one-to-one."""
    if horizon<=0 or len(set(onsets))!=len(onsets) or not np.isfinite(onsets).all(): raise ValueError('Invalid matching inputs')
    if len({f['completion_seconds'] for f in falls})!=len(falls): raise ValueError('Duplicate fall completion')
    used=set();matches=[]
    for fall in sorted(falls,key=lambda x:x['completion_seconds']):
        tf=fall['completion_seconds']
        eligible=[j for j,a in enumerate(onsets) if j not in used and tf-horizon-1e-9<=a<tf-1e-9]
        j=min(eligible,key=lambda j:onsets[j]) if eligible else None
        if j is not None: used.add(j)
        a=onsets[j] if j is not None else None
        ti=fall.get('impact_seconds')
        matches.append(dict(completion_seconds=tf,impact_seconds=ti,alarm_seconds=a,
                            completion_lead=tf-a if a is not None else None,
                            impact_lead=ti-a if ti is not None and a is not None else None))
    return matches,[a for j,a in enumerate(onsets) if j not in used]
