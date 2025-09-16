import math 
import torch
from torch import nn
from torch.nn import functional as F 

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):

        # BS, seq_len, Dim 

        input_shape = x.shape 
        batch_size, sequence_len, d_embed = input_shape

        intermim_shape = (batch_size, sequence_len, self.n_heads, self.d_head)

        # BS, seq_len, Dim -> Dim *3 -> 3 of Dim

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # BS, seq_len, Dim -> BS, seq_len, H, dim/H -> BS, H, seq_len, dim/H

        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # BS, H, seq_len, seq_len

        weight = q @ k.transpose(-1, -2)

        if causal_mask:

            # makes a triangle of ones, upper half
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)

        # BS, H, seq_len, seq_len @ BS, H, seq_len, Dim/H -> BS, H, seq_len, Dim/H 
        output = weight @ v 

        # BS, H, Seq_len, Dim/H
        output = output.transpose(1, 2)

        output = output.reshape(input_shape) # makes sure self_attn doesnt ruin convs

        # BS, seq_len, dim

        return output

class CrossAttention(nn.Module):

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True): 
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads 
        self.d_head = d_embed//n_heads

    def forward(self, x, y):

        # latent (x), bs, seq_len_q, dim_q 
        # content (y), bs, seq_len_kv, dim_kv = bs, 77, 768

        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape 

        intermim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by Wq matrix 

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(intermim_shape).transpose(1, 2) # bs, seq_len_kv, dim_kv -> bs, -1, n_heads, d_head -> bs, n_heads, -1, d_head?
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2) # k is now bs, n_heads, d_head, -1 ?
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v 
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output) 

        return output 

