"""Recompute metrics and detailed records from saved predictions without model weights."""
import _common
import argparse
import json
import math
import numpy as np
from pathlib import Path
from pft_et.utils.io import read_jsonl,save_json
from pft_et.metrics.evaluation import evaluate_predictions
from pft_et.metrics.audit import calibration,stream_audit
from pft_et.data.integrity import validate_records,valid_frames


def recompute(run,partition='test'):
    run=Path(run);context=json.loads((run/f'{partition}_evaluation_context.json').read_text())
    cfg=context['config'];predictions=read_jsonl(run/f'{partition}_predictions.jsonl')
    records=read_jsonl(run/f'{partition}_manifest.jsonl')
    validate_records(records)
    expected={(r['sequence_id'],f) for r in records for f in valid_frames(r,cfg)}
    if len(predictions)!=len(expected) or {(p['sequence_id'],p['frame']) for p in predictions}!=expected:
        raise ValueError('Predictions do not cover every protocol-valid position exactly once')
    metadata={r['sequence_id']:r for r in records};edges=np.rint(np.array(cfg['data']['bin_edges'])*cfg['data']['fps']).astype(int)
    for p in predictions:
        r=metadata[p['sequence_id']];tf=int(round(r['completion_seconds']*cfg['data']['fps'])) if r['fall'] else -1
        expected_label=int(np.searchsorted(edges[1:],tf-p['frame'],side='left')) if r['fall'] else len(edges)-1
        expected_tau=(tf-p['frame'])/cfg['data']['fps'] if r['fall'] else -1.
        if p['label']!=expected_label or abs(p['tau']-expected_tau)>1e-8:raise ValueError('Prediction labels/times disagree with manifest')
    actual,rows=evaluate_predictions(predictions,records,cfg,context['threshold'])
    saved=json.loads((run/f'{partition}_metrics.json').read_text());errors=[]
    for key in ('tp','fp','fn','event_f1','far','alt','adl_hours','adl_false_alarms','excluded_adl_seconds','preimpact_recall@0','preimpact_recall@0.25','preimpact_recall@0.5','preimpact_recall@1','event_time_nll','remaining_time_mae'):
        a,b=actual[key],saved[key]
        if not (a is None and b is None) and (a is None or b is None or not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-9)):
            errors.append(f'Metric mismatch: {key}')
    original={r['sequence_id']:r for r in read_jsonl(run/f'{partition}_events.jsonl')}
    metadata={r['sequence_id']:r for r in records}
    for r in rows:
        sid=r['sequence_id']; detail=stream_audit(r,[p for p in predictions if p['sequence_id']==sid],metadata[sid],cfg)
        for key,value in detail.items():
            if value!=original[sid].get(key): errors.append(f'Event/exposure trace mismatch: {sid}/{key}')
        valid=sum(x['end_frame_exclusive']-x['start_frame'] for x in detail['valid_intervals'])
        excluded=sum(x['end_frame_exclusive']-x['start_frame'] for group in detail['excluded_intervals_by_reason'].values() for x in group)
        if valid+excluded!=metadata[sid]['frames']: errors.append(f'Exposure reconciliation: {sid}')
    calculated=calibration(predictions,len(cfg['data']['bin_edges']))
    if calculated!=saved['calibration']: errors.append('Calibration mismatch')
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,n_prefixes=len(predictions),n_streams=len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--partition',default='test',choices=['val','test']);p.add_argument('--output',required=True)
    a=p.parse_args();report=recompute(a.run,a.partition);save_json(a.output,report);print(json.dumps(report,indent=2))
    raise SystemExit(bool(report['errors']))
