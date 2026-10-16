"""Content-addressed execution provenance without recording environment secrets."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from pft_et.utils.io import file_sha256


def execution_provenance(cfg, checkpoint=None):
    root = Path(__file__).resolve().parents[3]
    files = sorted(p for folder in ('src', 'scripts', 'configs')
                   for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml'))
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML', 'mmaction2', 'mmpose', 'mmcv'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    import torch
    return dict(code_sha256=digest, source_files=hashes, git_commit=commit,
                config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                python=sys.version.split()[0], platform=platform.platform(), packages=packages,
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None)
