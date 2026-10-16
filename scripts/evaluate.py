import _common
import argparse
import json
from pathlib import Path
from pft_et.data.dataset import PrefixDataset,make_loader,label_at
from pft_et.models.model import PFTET
from pft_et.training.prediction import predict
from pft_et.training.trainer import device_for
from pft_et.metrics.evaluation import evaluate_predictions,select_threshold
from pft_et.utils.checkpoint import load_checkpoint
from pft_et.utils.io import load_config,save_json,read_jsonl,write_jsonl,data_fingerprints,array_fingerprints
from pft_et.utils.seed import set_seed
from pft_et.metrics.audit import stream_audit,calibration
from pft_et.utils.provenance import execution_provenance
from pft_et.utils.compatibility import check_checkpoint_config

def evaluate(cfg,checkpoint,partition='test',predictions_file=None,far_budget=None,output=None,selected_threshold=None,analysis=False,validation_record=None):
    device=device_for(cfg);set_seed(cfg['training']['seed']+100000,cfg['training']['deterministic'])
    dataset=PrefixDataset(cfg,partition,include_future=analysis)
    directory=Path(output or cfg['experiment']['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    if (directory/f'{partition}_metrics.json').exists(): raise FileExistsError('Evaluation output exists; choose a new --output directory')
    if predictions_file:
        rows=read_jsonl(predictions_file)
        # Same exact positions must be evaluated for every external baseline.
        expected={(dataset.records[ri]['sequence_id'],t) for ri,t in dataset.index}
        actual={(r['sequence_id'],r['frame']) for r in rows}
        if expected!=actual or len(rows)!=len(expected): raise ValueError('Baseline predictions disagree with unified protocol positions')
        for r in rows:
            if abs(r['time']-r['frame']/cfg['data']['fps'])>1e-7: raise ValueError('Baseline timestamps disagree with frame grid')
            if r.get('risk') is None and r.get('probability') is None: raise ValueError('Baseline prediction requires risk or probability')
        threshold=selected_threshold
        if partition=='test' and threshold is None: raise ValueError('External test evaluation requires a validation-selected --threshold')
        if partition=='test':
            record_path=Path(validation_record) if validation_record else directory/'val_evaluation_context.json'
            if not record_path.is_file():raise ValueError('External test threshold requires --validation-record from a validation evaluation')
            valid_context=json.loads(record_path.read_text())
            if valid_context['partition']!='val' or abs(valid_context['threshold']-threshold)>1e-10:
                raise ValueError('Threshold does not match validation record')
            if valid_context['data_fingerprints']!=data_fingerprints(cfg) or valid_context['config']['training']['seed']!=cfg['training']['seed']:
                raise ValueError('Validation record uses another split or seed')
    else:
        ck=load_checkpoint(checkpoint,device=device)
        check_checkpoint_config(cfg,ck['config'])
        if ck['config']['training']['seed'] != cfg['training']['seed']:
            raise ValueError('Evaluation config seed must match the trained checkpoint seed')
        if ck.get('data_fingerprints') != data_fingerprints(cfg):
            raise ValueError('Data manifest or split bytes differ from the trained checkpoint')
        if ck.get('array_fingerprints') is not None and ck['array_fingerprints']!=array_fingerprints(cfg):raise ValueError('Array contents differ from training')
        # Test-time N_MC/marginalization may change; learned parameter shapes may not.
        for key in ('rgb_backend','dim','skeleton_layers','skeleton_heads','skeleton_ffn','mixtures','strategy'):
            if ck['config']['model'][key]!=cfg['model'][key]: raise ValueError(f'Checkpoint architecture mismatch: {key}')
        model=PFTET(cfg,initialize=False).to(device);model.load_state_dict(ck['model'],strict=True)
        rows=predict(model,make_loader(dataset,cfg),device,analysis)
        threshold=ck['threshold'] if selected_threshold is None else selected_threshold
    expected={(dataset.records[ri]['sequence_id'],t) for ri,t in dataset.index}
    actual={(r['sequence_id'],r['frame']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Predictions must cover every protocol-valid stream/frame exactly once')
    metadata={r['sequence_id']:r for r in dataset.records}
    for row in rows:
        if abs(row['time']-row['frame']/cfg['data']['fps'])>1e-7:
            raise ValueError('Prediction timestamp disagrees with unified frame grid')
        r=metadata[row['sequence_id']]
        tf=int(round(r['completion_seconds']*cfg['data']['fps'])) if r['fall'] else -1
        row['label']=label_at(row['frame'],tf,r['fall'],dataset.edges)
        row['tau']=(tf-row['frame'])/cfg['data']['fps'] if r['fall'] else -1.
    if partition=='val':
        selection=select_threshold(rows,dataset.records,cfg,far_budget)
        threshold=selection['threshold']
    elif far_budget is not None: raise ValueError('FAR budget selection uses validation only')
    metrics,events=evaluate_predictions(rows,dataset.records,cfg,float(threshold))
    for event in events:
        event.update(stream_audit(event,[p for p in rows if p['sequence_id']==event['sequence_id']],metadata[event['sequence_id']],cfg))
    metrics.update(partition=partition,seed=cfg['training']['seed'],synthetic=cfg['experiment']['synthetic'],
                   method=cfg['experiment']['name'],dataset=cfg['data']['dataset'],calibration=calibration(rows,len(cfg['data']['bin_edges'])))
    selection=selection['threshold_selection'] if partition=='val' else ck.get('threshold_selection') if not predictions_file else None
    if partition=='test' and predictions_file: selection=valid_context['selection']
    save_json(directory/f'{partition}_evaluation_context.json',dict(config=cfg,partition=partition,threshold=float(threshold),
              selection=selection,far_budget=far_budget,data_fingerprints=data_fingerprints(cfg),array_fingerprints=array_fingerprints(cfg),
              execution=execution_provenance(cfg,checkpoint),split=json.loads(Path(cfg['data']['splits']).read_text())))
    write_jsonl(directory/f'{partition}_manifest.jsonl',dataset.records)
    write_jsonl(directory/f'{partition}_predictions.jsonl',rows)
    write_jsonl(directory/f'{partition}_events.jsonl',events)
    save_json(directory/f'{partition}_metrics.json',metrics)
    print(json.dumps({k:metrics[k] for k in ['event_f1','far','alt','threshold','event_time_nll']},indent=2))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--checkpoint');p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--predictions');p.add_argument('--far-budget',type=float);p.add_argument('--threshold',type=float);p.add_argument('--output');p.add_argument('--analysis',action='store_true');p.add_argument('--validation-record')
    a=p.parse_args()
    if not a.checkpoint and not a.predictions: p.error('Supply --checkpoint or --predictions')
    evaluate(load_config(a.config),a.checkpoint,a.partition,a.predictions,a.far_budget,a.output,a.threshold,a.analysis,a.validation_record)
