# utils.py
import torch
import numpy as np
import random
import os

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

CLEAN_MODEL_DIR = "model_weights"