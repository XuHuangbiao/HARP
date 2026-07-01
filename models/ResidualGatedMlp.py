import math

import torch
from torch import nn
import torch.nn.functional as F

from models.AttnW import FeatDimAttnW
from models.DyT import DyT
from utils.misc import cnt_kel

class HTPB(nn.Module):
    def __init__(self, args, n_node, d_model, dropout, temp, softmax_dim):
        super(HTPB, self).__init__()
        self.head_dim = d_model // n_node
        self.n_node = n_node
        ffn_fac = args.models['ffn_fac']
        ffn_dim = d_model * ffn_fac
        self.args = args
        self.attn = FeatDimAttnW(d_model=d_model, nhead=n_node, dropout=dropout,
                                 ffn_fac=ffn_fac, temp=temp, softmax_dim=softmax_dim)
        self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.shape[0]
        attn_outs, attn = self.attn(x)  

        ffn_outs = self.ffn(attn_outs.view(bs, -1))

        leaf_fused = self.dropout(ffn_outs) + attn_outs.view(bs, -1)
        leaf_fused_reshape = leaf_fused
        return leaf_fused_reshape, attn
        


    
class ResidualGatedMlp(nn.Module):
    def __init__(self, args, dim_input=256, in_dae=256, dropout=0.2, hid_factor=1):
        super(ResidualGatedMlp, self).__init__()
        self.args = args
        self.hidden_size = dim_input
        self.dropout = nn.Dropout(dropout)
        self.x_proj = nn.Linear(dim_input, self.hidden_size)
        self.uni_proj = nn.Linear(dim_input, self.hidden_size)
        self.tree_out_norm = DyT(dim_input)
        self.uni_norm = DyT(self.hidden_size)
        self.x_norm = DyT(self.hidden_size)
        self.fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.temperature = 0.07

        
        self.node_array = args.models['node_array']
        self.num_levels = len(self.node_array)

        
        for n_nodes in self.node_array:
            assert self.hidden_size % n_nodes == 0, \
                f"hidden_size ({self.hidden_size}) must be divisible by all node counts in {self.node_array}"

        
        self.tree_layers = nn.ModuleList([
            HTPB(
                args=args,
                n_node=n_nodes,
                d_model=self.hidden_size,
                dropout=dropout,
                temp=self.temperature,
                softmax_dim=-1
            ) for n_nodes in self.node_array
        ])

        for i, layer in enumerate(self.tree_layers):
            layer.num_leaf = self.node_array[i]

        self.mlp = nn.Sequential(
            DyT(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, dim_input),
        )

        self.forget_gate = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            DyT(dim_input),
            nn.Linear(dim_input, in_dae),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, uni_feat):
        input_feat = self.x_norm(self.x_proj(x))
        if uni_feat is not None:
            fused_feat = self.fusion(torch.cat([input_feat, uni_feat], dim=-1))
            input_feat = input_feat + fused_feat

        
        current_feat = input_feat
        orth_list = []
        for tree_layer in self.tree_layers:
            fused_out, orth = tree_layer(current_feat)
            residual_scale = torch.sigmoid(current_feat.var(dim=-1, keepdim=True))
            current_feat = fused_out + residual_scale * current_feat
            orth_list.append(orth)

        
        mlp_out = self.mlp(current_feat) + current_feat
        mlp_out = self.tree_out_norm(mlp_out)

        
        forget_weight = self.forget_gate(x)
        
        x_remains = x * forget_weight
        
        residual_out = x_remains + mlp_out
        
        output = self.out(residual_out)

        return output, orth_list