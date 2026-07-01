import torch
from torch import nn
from .DAE import DAE
from .DyT import DyT
from .transfomer import PairModel
from .uniform import UniformModule

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pair_model = PairModel(args)
        self.uniform = UniformModule(args=args, dropout=args.models['dropout_rate'])
        self.dae = DAE(args=args, in_dae=args.models['in_dae'], dropout=args.models['dropout_rate'])

    def forward(self, feature):
        uni_feat, uni_pred = self.uniform(feature)
        x, orth_list = self.pair_model(feature, uni_feat)
        pred, mu, std = self.dae(x)
        return pred, mu, std, uni_pred, orth_list

