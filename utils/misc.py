import argparse
import math
import os
import random

import numpy as np
from pydoc import locate

import torch
from torch import nn


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def import_model(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(name):
    return locate(name)



def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def use_DataParallel(args, model):
    return nn.DataParallel(model, device_ids=args.device, output_device=args.output_device)

def get_sinusoid_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def cnt_kel(base):
    kernel_size = max(1, base - 1) if base % 2 == 0 else base
    padding = (kernel_size - 1) // 2
    return kernel_size, padding