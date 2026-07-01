import torch
import torch.nn as nn

from models.AttnW import SeqAttnW
from models.DyT import DyT
from utils.misc import cnt_kel

class UniformModule(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(UniformModule, self).__init__()
        self.args = args
        num_cls = args.models['num_cls']
        self.target_dim = args.models['d_model']
        self.input_dim = args.models['in_features']
        self.fc_output = 128
        self.seq = args.models['seq_len']
        kel, pad = cnt_kel(num_cls)
        
        self.proj = nn.Conv1d(self.input_dim, self.target_dim, kernel_size=kel, padding=pad)

        self.attn = SeqAttnW(seq=self.seq, input_dim=self.target_dim, hidden_dim=self.fc_output, target_dim=1, dropout=dropout, temp=1.2)

        
        self.attn_proj = nn.Linear(self.target_dim, self.target_dim)
        self.attn_norm = DyT(self.target_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.uniform_pred = nn.Sequential(
            DyT(self.target_dim),
            nn.Linear(self.target_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        
        

        x = x.transpose(1, 2)  
        x = self.proj(x)  
        x = x.transpose(1, 2)  

        attn_output = self.attn(x)  

        
        pooled_features = torch.sum(attn_output, dim=1, keepdim=False)  

        
        attn_output = self.attn_proj(pooled_features)  
        attn_output = self.attn_norm(attn_output)
        attn_output = self.attn_dropout(attn_output)

        if hasattr(self, 'uniform_pred') and self.uniform_pred is not None:
            uni_pred = self.uniform_pred(attn_output).squeeze(-1)
        else:
            uni_pred = None

        return attn_output, uni_pred