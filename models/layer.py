import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.AttnW import SeqAttnW
from models.DyT import DyT
from utils.misc import get_sinusoid_encoding, cnt_kel


class Attention(nn.Module):
    def __init__(self, d_model, nhead):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.temp = nn.Parameter(torch.tensor(0.7))
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.last_attn = None 

    def forward(self, q, k, v):
        bs, seq_len, _ = q.size()
        q = q.view(bs, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(bs, -1, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores / self.temp, dim=-1)  
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)

        self.last_attn = attn.detach()  

        return out

class TransformerLayer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, ffn_factor=4, dropout=0.1, num_cls=5, use_self_attn=True):
        super(TransformerLayer, self).__init__()
        kel, pad = cnt_kel(num_cls)
        self.use_self_attn = use_self_attn
        self.q_linear_cross = nn.Conv1d(d_model, d_model, kernel_size=kel, padding=pad)
        self.k_linear_cross = nn.Conv1d(in_channels, d_model, kernel_size=kel, padding=pad)
        self.v_linear_cross = nn.Conv1d(in_channels, d_model, kernel_size=kel, padding=pad)

        self.q_linear_self = nn.Conv1d(d_model, d_model, kernel_size=kel, padding=pad)
        self.k_linear_self = nn.Conv1d(d_model, d_model, kernel_size=kel, padding=pad)
        self.v_linear_self = nn.Conv1d(d_model, d_model, kernel_size=kel, padding=pad)

        self.cross_attn = Attention(d_model, nhead)
        self.self_attn = Attention(d_model, nhead)
        self.norm_cross = DyT(d_model)
        self.norm_self = DyT(d_model)
        self.norm_ffn1 = DyT(d_model)
        self.norm_ffn2 = DyT(d_model)

        self.dropout = nn.Dropout(dropout)
        ffn_dim = int(d_model * ffn_factor)
        self.ffn1 = nn.Sequential(  
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.ffn2 = nn.Sequential(  
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, query, memory, query_embed):
        
        
        q_cross = self.norm_cross(self.with_pos_embed(query, query_embed))  
        q_cross = self.q_linear_cross(q_cross.transpose(1, 2)).transpose(1, 2)  
        memory_t = memory.transpose(1, 2)
        k_cross = self.k_linear_cross(memory_t).transpose(1, 2)  
        v_cross = self.v_linear_cross(memory_t).transpose(1, 2)  
        cross_out = self.cross_attn(q_cross, k_cross, v_cross)
        query = query + self.dropout(cross_out)

        
        q_ffn1 = self.norm_ffn1(query)  
        ffn1_out = self.ffn1(q_ffn1)  
        query = query + self.dropout(ffn1_out)

        if self.use_self_attn:
            
            query = self.norm_self(query)
            query_t = query.transpose(1, 2)  
            q_self = self.q_linear_self(query_t).transpose(1, 2)
            k_self = self.k_linear_self(query_t).transpose(1, 2)
            v_self = self.v_linear_self(query_t).transpose(1, 2)
            self_out = self.self_attn(q_self, k_self, v_self)
            query = query + self.dropout(self_out)

            
            q_ffn2 = self.norm_ffn2(query)  
            ffn2_out = self.ffn2(q_ffn2)  
            query = query + self.dropout(ffn2_out)

        return query

class SelfAttnAggregator(nn.Module):
    def __init__(self, d_model, num_cls, dropout=0.1):
        super(SelfAttnAggregator, self).__init__()
        self.attn = SeqAttnW(seq=num_cls, input_dim=d_model, hidden_dim=128, target_dim=1,
                             dropout=dropout, softmax_dim=1, temp=1.0)
        self.norm = DyT(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_tokens):
        
        attn = self.attn(cls_tokens)
        x = torch.sum(attn, dim=1, keepdim=False)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class HybridTransformer(nn.Module):
    def __init__(self, in_channels=1024, d_model=512, nhead=4, num_layers=2,
                 ffn_factor=2, dropout=0.2, seq_len=400, num_cls=5):
        super(HybridTransformer, self).__init__()
        
        self.query_embed = nn.Embedding(num_cls, d_model)
        self.pos_embedding = nn.Parameter(get_sinusoid_encoding(seq_len, in_channels))
        self.num_cls = num_cls
        self.layers = nn.ModuleList([
            TransformerLayer(in_channels, d_model, nhead, ffn_factor, dropout, num_cls, True)  
            for i in range(num_layers)
        ])
        self.aggregator = SelfAttnAggregator(d_model, num_cls, dropout)
        self.norm = DyT(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        
        bs = x.shape[0]
        memory = x + self.pos_embedding.repeat(bs, 1, 1).to(x.device)
        
        cls_ids = torch.arange(self.num_cls, device=x.device)
        query_embed = self.query_embed(cls_ids).unsqueeze(0).repeat(bs, 1, 1)
        query = torch.zeros_like(query_embed)

        for layer in self.layers:
            query = layer(query, memory, query_embed)

        global_feature = self.aggregator(query)
        return global_feature


if __name__ == '__main__':
    batch_size = 32
    seq_len = 400
    d_fea = 1024
    v1 = torch.randn(batch_size, seq_len, d_fea)
    model = HybridTransformer(in_channels=d_fea, d_model=512, nhead=16, num_layers=2,
                              ffn_factor=4, dropout=0.1, seq_len=seq_len, num_cls=1)
    x1 = model(v1)
    print(f"x1 shape: {x1.shape}")  