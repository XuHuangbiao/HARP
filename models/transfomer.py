from .layer import *
from .ResidualGatedMlp import ResidualGatedMlp
class PairModel(nn.Module):
    def __init__(self, args):
        self.args = args
        super(PairModel, self).__init__()
        self.transformer = HybridTransformer(in_channels=args.models['in_features'], d_model=args.models['d_model'],
                                             nhead=args.models['num_head'], num_layers=args.models['nlayer'],
                                             num_cls=args.models['num_cls'], dropout=args.models['dropout_rate'],
                                             ffn_factor=args.models['ffn_fac'])
        self.res_mlp = ResidualGatedMlp(args=args, dim_input=args.models['d_model'], hid_factor=args.models['res_hid_fac'],
                                        in_dae=args.models['in_dae'], dropout=args.models['dropout_rate'])


    def forward(self, x, uni_fea):
        trans_feat = self.transformer(x)

        x, orth_list = self.res_mlp(trans_feat, uni_fea)

        return x, orth_list


