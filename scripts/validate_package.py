"""Validate repository structure, syntax, documentation links, and optional release hashes."""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path


def validate_reported_tables(folder):
    """Check an optional external table catalogue without hard-coded research results."""
    folder=Path(folder);errors=[];checks=[]
    index=folder/'TABLE_INDEX.json'
    if not index.is_file():return dict(errors=['Missing optional table catalogue'],checks=checks)
    try:
        entries=json.loads(index.read_text(encoding='utf-8'))
        for entry in entries:
            path=folder/Path(entry['file']).name
            if not path.is_file():errors.append('Missing table: '+path.name);continue
            with path.open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f);rows=list(reader)
            if len(rows)!=entry['rows']:errors.append('Table row count mismatch: '+path.name)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                errors.append('Inconsistent CSV fields: '+path.name)
        checks.append(str(len(entries))+' optional table files checked')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        errors.append('Invalid optional table catalogue: '+str(exc))
    return dict(errors=errors,checks=checks)


def validate(root,check_index=False,reported_tables=None):
    root=Path(root).resolve();errors=[];checks=[];warnings=[]
    required=['README.md','LICENSE.md','CITATION.cff','pyproject.toml','requirements.txt',
              'configs/default.yaml','src/pft_et/models/model.py','scripts/train.py','scripts/evaluate.py',
              'scripts/build_manifest.py','scripts/summarize_dataset.py','scripts/validate_research_inputs.py',
              'scripts/audit_baseline_runs.py','examples/README.md','examples/data/metadata.example.csv',
              'examples/data/splits.example.json','examples/evaluation/external_predictions.example.jsonl']
    for name in required:
        if not (root/name).is_file():errors.append('Missing repository file: '+name)
    files=sorted((root/'src').rglob('*.py'))+sorted((root/'scripts').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    for path in files:
        try:ast.parse(path.read_text(encoding='utf-8'),filename=str(path.relative_to(root)))
        except (SyntaxError,UnicodeError) as exc:errors.append('Python syntax: '+str(exc))
    checks.append(str(len(files))+' Python files parsed')
    readme=root/'README.md'
    if readme.is_file():
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',readme.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
            target=target.split('#',1)[0]
            if target and not (root/target).exists():errors.append('Missing README link: '+target)
        checks.append('README local links checked')
    examples=root/'examples'
    if examples.is_dir():
        for path in sorted(examples.rglob('*.jsonl')):
            try:
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                if not rows:errors.append('Empty JSONL example: '+str(path.relative_to(root)))
            except (ValueError,OSError) as exc:errors.append('Invalid JSONL example: '+str(exc))
        for path in sorted(examples.rglob('*.json')):
            try:json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError) as exc:errors.append('Invalid JSON example: '+str(exc))
        checks.append('example JSON records parsed')
    forbidden=[]
    for path in root.rglob('*'):
        if path.is_file() and (path.suffix in ('.pyc','.pyo') or '__pycache__' in path.parts):forbidden.append(path)
    if forbidden:warnings.append(str(len(forbidden))+' local cache files are excluded by the release packager')
    if check_index:
        index=root/'docs/FILE_INDEX.csv';seen=set()
        if not index.is_file():errors.append('Missing release file index')
        else:
            with index.open(encoding='utf-8',newline='') as f:
                for row in csv.DictReader(f):
                    rel=Path(row['path']);path=(root/rel).resolve()
                    if rel.is_absolute() or not path.is_relative_to(root):
                        errors.append('Invalid indexed path: '+row['path']);continue
                    if row['path'] in seen:errors.append('Duplicate indexed path: '+row['path'])
                    seen.add(row['path'])
                    if not path.is_file():errors.append('Missing indexed file: '+row['path']);continue
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                        errors.append('SHA256 mismatch: '+row['path'])
            checks.append(str(len(seen))+' indexed file hashes checked')
    if reported_tables is not None:
        optional=validate_reported_tables(reported_tables);errors.extend(optional['errors']);checks.extend(optional['checks'])
    return dict(overall='PASS' if not errors else 'ERROR',errors=errors,warnings=warnings,checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output')
    p.add_argument('--check-index',action='store_true',help='Check a frozen release inventory; refresh it after source edits')
    p.add_argument('--reported-tables',help='Optional external folder containing TABLE_INDEX.json and CSVs')
    a=p.parse_args();report=validate(a.root,a.check_index,a.reported_tables)
    if a.output:
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(bool(report['errors']))
