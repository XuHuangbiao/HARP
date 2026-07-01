import random

import torch
from torchvision import transforms

class AddNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = torch.normal(mean=self.mean, std=self.std, size=x.shape, device=x.device)
        return x + noise


class BlockClipScaling(object):
    def __init__(self, scale_range=(0.9, 1.1), block_size=10):
        self.scale_range = scale_range
        self.block_size = block_size

    def __call__(self, x):
        if self.scale_range[0] == self.scale_range[1] == 1.0:
            return x
        T, _ = x.size()
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        x_scaled = x * scale
        return x_scaled


class RandomClipDropout(object):
    def __init__(self, dropout_prob=0.15):
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        if self.dropout_prob <= 0.0:
            return x

        seq_len, dim_feat = x.shape
        mean = 0.2576
        

        mask = torch.rand(seq_len, dim_feat) < self.dropout_prob
        x_drop = x.clone()
        x_drop[mask] = mean

        return x_drop

class RandomTemporalReverse(object):
    def __init__(self, reverse_prob=0.2):
        self.reverse_prob = reverse_prob
    def __call__(self, x):
        if random.random() < self.reverse_prob:
            x = torch.flip(x, dims=[0])
            x = torch.flip(x, dims=[1])
        return x


def get_feature_trans(args):
    train_trans = transforms.Compose([
        AddNoise(mean=args.data_aug['noise_mean'], std=args.data_aug['noise_std']),
        
    ])

    test_trans = transforms.Compose([
    ])

    return train_trans, test_trans