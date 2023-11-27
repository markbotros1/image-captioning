import torch.nn as nn
from torch.nn import Transformer as T

class MultiHeadAttention(nn.Module):
    def __init__(self, n_emb, n_head, dropout=0.0, bias=False, cross_attn=False):
        super().__init__()
        self.n_emb = n_emb
        if not cross_attn:
            self.qkv_proj = nn.Linear(n_emb, 3*n_emb, bias=bias)
        else:
            self.q_proj = nn.Linear(n_emb, n_emb, bias=bias)
            self.kv_proj = nn.Linear(n_emb, 2*n_emb, bias=bias)
        self.msa = nn.MultiheadAttention(n_emb, n_head, 
                                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.att_proj = nn.Linear(n_emb, n_emb, bias=bias)

    def forward(self, x, enc_out=None):
        if enc_out == None:
            q, k, v  = self.qkv_proj(x).split(self.n_emb, dim=-1)
        else:
            q = self.q_proj(x)
            k, v = self.kv_proj(enc_out).split(self.n_emb, dim=-1)
        x = self.msa(q, k, v, need_weights=False, is_causal=False)
        x = self.att_proj(self.dropout(x[0]))
        return x
    

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, n_emb, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.n_emb = n_emb
        self.qkv_proj = nn.Linear(n_emb, 3*n_emb, bias=bias)
        self.msa = nn.MultiheadAttention(n_emb, n_head, 
                                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.att_proj = nn.Linear(n_emb, n_emb, bias=bias)

    def forward(self, x, pad_mask, device):
        b, t, c = x.size()
        mask = T.generate_square_subsequent_mask(t).to(device)
        q, k, v  = self.qkv_proj(x).split(self.n_emb, dim=-1)
        if pad_mask != None:
            x = self.msa(q, k, v, key_padding_mask=pad_mask, need_weights=False, 
                        attn_mask=mask, is_causal=True)
        else:
            x = self.msa(q, k, v, need_weights=False, attn_mask=mask, is_causal=True)
        x = self.att_proj(self.dropout(x[0]))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_emb, dropout=0.0, bias=False):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb, bias=bias),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffwd(x)