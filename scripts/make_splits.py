import _common
import argparse
from pft_et.data.splits import generate_split,validate_splits
from pft_et.utils.io import read_jsonl,save_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Create a protocol-consistent alternative split; original IDs must be supplied separately.')
    p.add_argument('--manifest',required=True);p.add_argument('--dataset',choices=['upfall','urfall'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();save_json(a.output,generate_split(read_jsonl(a.manifest),a.dataset,a.seed))
