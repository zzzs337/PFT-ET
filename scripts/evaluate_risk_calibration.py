"""Brier and ten-bin ECE using the same implementation as the main evaluator."""
import _common
import argparse
from pathlib import Path
from pft_et.metrics.audit import calibration
from pft_et.utils.io import read_jsonl,save_json


def compute_calibration(rows,bins=10,states=7):
    return calibration(rows,states,bins)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--bins',type=int,default=10);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Choose a new calibration output')
    result=compute_calibration(read_jsonl(a.predictions),a.bins);save_json(a.output,result)
    print(f"Brier={result['brier_score']:.6f}; ECE={result['ece']:.6f}; N={result['n_prefixes']}")
