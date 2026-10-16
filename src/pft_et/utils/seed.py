import random
import numpy as np
import torch

def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)

def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)
