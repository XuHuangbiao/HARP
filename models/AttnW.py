import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DyT import DyT
from utils.misc import get_sinusoid_encoding, cnt_kel

class PEG(nn.Module):
    """Conditional Positional Encoding"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        kernel_size, padding = cnt_kel(kernel_size)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        
        x = x.transpose(1, 2)  
        pos_enc = self.conv(x)
        out = (x + pos_enc).transpose(1, 2)
        return out

class SeqAttnW(nn.Module):
    def __init__(self, seq, input_dim=256, hidden_dim=256, target_dim=256, dropout=0.1, temp=None, softmax_dim=1):
        super(SeqAttnW, self).__init__()
        self.d_model = input_dim
        self.peg = PEG(input_dim, kernel_size=7)
        self.seq = seq
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.norm = DyT(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.dim = softmax_dim
        self.temp = nn.Parameter(torch.ones([]) * temp) if temp is not None else torch.tensor(1.0)
        self.residual_weight = nn.Parameter(torch.ones(1))
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.peg(x)
        x_norm = self.norm(x)  
        mlp_out = self.mlp(x_norm) / self.temp
        head_weights = F.softmax(mlp_out, dim=self.dim)
        attn = x_norm * head_weights
        
        residual = torch.sigmoid(x_norm.var(dim=-1, keepdim=True)) * x_norm
        attn = attn + residual
        return attn

class FeatDimAttnW(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.2, temp=None, ffn_fac=2, softmax_dim=-1):
        super(FeatDimAttnW, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        hid_dim = ffn_fac * self.head_dim
        kernel_size, padding = cnt_kel(nhead)
        self.peg = PEG(self.head_dim, kernel_size=kernel_size)
        self.head_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, hid_dim),
                nn.GELU(),
                nn.Linear(hid_dim, hid_dim),
                nn.GELU(),
                nn.Linear(hid_dim, self.head_dim),
            ) for _ in range(nhead)
        ])

        self.clp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size, padding=padding),
        )
        self.pre_norm = DyT(self.head_dim)
        self.conv_norm = DyT(self.head_dim)
        self.post_norm = DyT(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.temp = nn.Parameter(torch.ones([]) * temp) if temp is not None else torch.tensor(1.0)
        self.softmax_dim = softmax_dim
        self.residual_weight = nn.Parameter(torch.ones(nhead, 1))
        self.gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        bs, feat_dim = x.shape  
        x = x.view(bs, self.nhead, self.head_dim)
        x_pos = self.peg(x)
        x_norm = self.pre_norm(x)
        conv_out = self.conv_norm(
            x_norm + self.conv(x_pos.transpose(1, 2)).transpose(1, 2)
        )  

        head_weights = []
        for i, attn_layer in enumerate(self.head_attn):
            head_input = x_norm[:, i, :]
            conv_in = conv_out[:, i, :]
            conv_cxt = self.clp(conv_in)

            mlp_attn_out = attn_layer(head_input + conv_cxt)  
            weights = mlp_attn_out

            weights = F.softmax(weights / self.temp, dim=self.softmax_dim)  
            head_weights.append(weights)

        head_weights = torch.stack(head_weights, dim=1)  
        output = conv_out * head_weights  

        
        residual = torch.sigmoid(x_norm.var(dim=-1, keepdim=True)) * x_norm
        output = residual + self.dropout(output)  
        attn = self.post_norm(output)  
        return attn, output