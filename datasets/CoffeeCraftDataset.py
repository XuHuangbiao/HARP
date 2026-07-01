import os
import sys
import torch
import torch.utils.data as data
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

class FeatureRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path_better(self):
        return self._data[0]

    @property
    def path_worse(self):
        return self._data[1]

class CoffeeCraftDataset(data.Dataset):
    def __init__(self, args, subset, transform=None, ftr_tmpl='{}.npz'):
        self.root_path = args.feature_path
        
        self.pair_list_path = args.train_pair_path if subset == 'train' else args.test_pair_path
        self.ftr_tmpl = ftr_tmpl
        self.transform = transform
        self._parse_list()
        self.subset = subset

    def _load_features(self, vid):
        features = np.load(os.path.join(self.root_path, self.ftr_tmpl.format(vid)))['features'].astype(np.float32)
        return features

    def _parse_list(self):
        self.pair_list = [FeatureRecord(x.strip().split(' ')) for x in open(self.pair_list_path)]

    def __getitem__(self, index):
        record = self.pair_list[index]
        vid1, vid2 = self.get_features(record)
        if self.transform is not None:
            vid1 = self.transform(torch.tensor(vid1))
            vid2 = self.transform(torch.tensor(vid2))
        return (vid1, vid2), (record.path_better, record.path_worse)

    def get_features(self, record):
        vid1 = self._load_features(record.path_better)
        vid2 = self._load_features(record.path_worse)
        return vid1, vid2

    def __len__(self):
        return len(self.pair_list)