import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def mod(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_repository_integrity_without_optional_tables():
    assert not mod('validate_package').validate(ROOT,check_index=False)['errors']


def test_optional_table_catalogue(tmp_path):
    import json
    validator=mod('validate_package')
    (tmp_path/'values.csv').write_text('metric,value\na,0.5\n')
    index=tmp_path/'TABLE_INDEX.json'
    index.write_text(json.dumps([dict(file='data/reported_results/values.csv',rows=1)]))
    assert not validator.validate_reported_tables(tmp_path)['errors']
    index.write_text(json.dumps([dict(file='data/reported_results/values.csv',rows=2)]))
    assert validator.validate_reported_tables(tmp_path)['errors']


def test_release_hash_check_is_explicit(tmp_path):
    import hashlib
    validator=mod('validate_package');(tmp_path/'docs').mkdir()
    (tmp_path/'README.md').write_text('repository\n')
    digest=hashlib.sha256((tmp_path/'README.md').read_bytes()).hexdigest()
    (tmp_path/'docs/FILE_INDEX.csv').write_text('path,bytes,sha256\nREADME.md,11,'+digest+'\n')
    (tmp_path/'README.md').write_text('edited\n')
    assert not any('SHA256' in e for e in validator.validate(tmp_path)['errors'])
    assert any('SHA256' in e for e in validator.validate(tmp_path,check_index=True)['errors'])


def test_annotation_csv_and_dataset_summary_examples(tmp_path):
    builder=mod('build_manifest');summarizer=mod('summarize_dataset')
    records=builder.convert(ROOT/'examples/data/metadata.example.csv')
    assert len(records)==6
    assert len({r['physical_id'] for r in records})==3
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    result=summarizer.summarize(manifest,ROOT/'examples/data/splits.example.json','upfall',tmp_path/'audit')
    assert result['fall_physical_events']==2
    assert (tmp_path/'audit/event_camera_map.csv').is_file()


def test_baseline_audit_reports_missing_evidence(tmp_path):
    report=mod('audit_baseline_runs').audit(ROOT/'configs/baselines.yaml',tmp_path)
    assert report['complete'] is False
    assert set(report['methods'])=={'tfd','tcnte','pose_vit','dgnn_lstm','dynamic_deephit'}


def test_pretrained_readme_has_no_unavailable_verification_reference():
    text=(ROOT/'pretrained/README.md').read_text()
    assert 'docs/verification/backbone.json' not in text


def test_risk_calibration_perfect_and_bounds():
    run=mod('evaluate_risk_calibration')
    rows=[dict(sequence_id='a',frame=1,label=0,probability=[1,0,0,0,0,0,0]),
          dict(sequence_id='b',frame=1,label=6,probability=[0,0,0,0,0,0,1])]
    result=run.compute_calibration(rows)
    assert result['brier_score']==0 and result['ece']==0
    with pytest.raises(ValueError):run.compute_calibration(rows+rows[:1])


def test_shared_transition_deterministic():
    fn=mod('evaluate_shared_transition').calculate
    targets={('s',10):dict(target=[1.,2.])}
    predictions={('s',10):dict(mean=[1.,1.])}
    assert np.isclose(fn(targets,predictions)['transition_rmse'],np.sqrt(.5))
    assert fn(targets,predictions)['transition_nll'] is None


def test_shared_transition_gaussian_coverage():
    fn=mod('evaluate_shared_transition').calculate
    targets={('s',10):dict(target=[0.,0.])}
    predictions={('s',10):dict(pi=[1.],mu=[[0.,0.]],sigma=[[1.,1.]])}
    result=fn(targets,predictions)
    assert np.isclose(result['transition_nll'],np.log(2*np.pi))
    assert result['coordinate_coverage']['0.9']['empirical']==1.0
    assert np.isclose(result['coordinate_coverage']['0.9']['mean_width'],3.2897072539,atol=1e-5)
