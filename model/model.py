import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_M_Weights, efficientnet_v2_m
from torchvision.models.feature_extraction import create_feature_extractor

import einops as E

from .blocks import Encoder, Decoder

class ImageCaptionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # EfficientNetV2 (CNN) Backbone
        nodes = {'features.8': 'features'}
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        self.cnn = efficientnet_v2_m(weights=weights)
        self.feat_extract = create_feature_extractor(self.cnn, return_nodes=nodes)
        # Transformer Encoder
        self.input_proj = nn.Linear(config.input_size, config.n_emb)
        self.spatial_emb = nn.Embedding(config.enc_block_size, config.n_emb)
        self.encoder = nn.Sequential(*[Encoder(config) for _ in range(config.n_layer)])
        # Transformer Decoder
        self.token_emb = nn.Embedding(config.vocab_size, config.n_emb, 
                                      padding_idx=config.padding_idx)
        self.position_emb = nn.Embedding(config.dec_block_size, config.n_emb)
        self.decoder = nn.Sequential(*[Decoder(config) for _ in range(config.n_layer)])
        # LayerNorm and Logits
        self.ln_f = nn.LayerNorm(config.n_emb)
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)

        self.start_idx = config.start_idx
        self.dev = torch.device(config.device) if config.device == 'cpu' else 'mps:0'

    def forward(self, cnn_in, dec_in, pad_masks=None):
        # Extract batch features from CNN
        self.cnn.eval()
        with torch.no_grad():
            features = self.feat_extract(cnn_in)['features']
        enc_in = E.rearrange(features, 'b c h w -> b (h w) c')
        # Encode
        _, T, _ = enc_in.shape
        enc_in = self.input_proj(enc_in)
        enc_in = enc_in + self.spatial_emb(torch.arange(T, device=self.dev))
        enc_out = self.encoder(enc_in)
        # Decode
        _, T = dec_in.shape
        dec_in = self.token_emb(dec_in)
        dec_in = dec_in + self.position_emb(torch.arange(T, device=self.dev))
        dec_out = self.decoder((dec_in, enc_out, pad_masks, self.dev))
        return self.lm_head(self.ln_f(dec_out[0]))

    def freeze_cnn(self):
        for name, param in self.named_parameters():            
            if name.startswith('cnn'):
                param.requires_grad = False

    def get_num_params(self, trainable=True):
        if trainable:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            n = sum(p.numel() for p in self.parameters())
        return n

    def generate_caption(self, cnn_in, block_size, max_tokens=500):
        idx = torch.LongTensor([[self.start_idx]])
        for _ in range(max_tokens):
            _, T = idx.shape
            if T > block_size:
                idx = idx[:, -block_size:]
            logits = self(cnn_in, idx)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx.to('cpu'), idx_next.to('cpu')), dim=-1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer
        extra_args = dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config(cnf_name):
    with open(cnf_name) as file:
        config = yaml.safe_load(file)
    return config