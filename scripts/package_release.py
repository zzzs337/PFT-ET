"""Create and integrity-check a portable source archive without runtime caches."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from release_files import selected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Release archive already exists')
    if out.is_relative_to(root):raise ValueError('Archive must be outside the source tree')
    out.parent.mkdir(parents=True,exist_ok=True)
    files=list(selected(root))
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,Path('PFT-ET')/f.relative_to(root))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise RuntimeError('Archive integrity failure')
    print(json.dumps(dict(files=len(files),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
