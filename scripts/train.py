import _common
import argparse
from pft_et.utils.io import load_config
from pft_et.training.trainer import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--resume')
    a=p.parse_args();train(load_config(a.config),a.resume)
