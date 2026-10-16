"""Create a content inventory of source, configuration, tests, and documentation."""
import argparse
import csv
import hashlib
from pathlib import Path
from release_files import selected


def build(root):
    root=Path(root).resolve();destination=root/'docs/FILE_INDEX.csv';rows=[]
    for path in selected(root):
        if path==destination:continue
        rel=path.relative_to(root)
        if rel.parts[:2]==('docs','verification'):continue
        rows.append(dict(path=rel.as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with destination.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();print(build(a.root))
