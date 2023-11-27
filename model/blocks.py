import torch.nn as nn
from .layers import MultiHeadAttention, CausalMultiHeadAttention, FeedForward

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb, bias=config.bias)
        self.msa = MultiHeadAttention(config.n_emb, 
                                      config.n_head, 
                                      dropout=config.dropout, 
                                      bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_emb, bias=config.bias)
        self.ffwd = FeedForward(config.n_emb, 
                                dropout=config.dropout, 
                                bias=config.bias)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb, bias=config.bias)
        self.causal_msa = CausalMultiHeadAttention(config.n_emb, 
                                                   config.n_head, 
                                                   dropout=config.dropout, 
                                                   bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_emb, bias=config.bias)
        self.cross_msa = MultiHeadAttention(config.n_emb, 
                                            config.n_head, 
                                            dropout=config.dropout, 
                                            bias=config.bias, 
                                            cross_attn=True)
        self.ln3 = nn.LayerNorm(config.n_emb, bias=config.bias)
        self.ffwd = FeedForward(config.n_emb, 
                                dropout=config.dropout, 
                                bias=config.bias)

    def forward(self, input):
        dec_in, enc_out, pad_masks = input[0], input[1], input[2]
        device = input[3]
        x = dec_in + self.causal_msa(self.ln1(dec_in), pad_masks, device)
        x = x + self.cross_msa(self.ln2(x), enc_out)
        x = x + self.ffwd(self.ln3(x))
        return (x, enc_out, pad_masks, device)
