import torch
from torch import nn
from torch.nn import functional as F 

from attention import SelfAttention

# defining clip to use it with the SD model

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab, n_embd, seq_len):
        super.__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.pos_embedding = nn.Parameters(torch.zeros(seq_len, n_embd)) # oh, so like rather than use rope or sinusoidal, they use defined pos

    def forward(self, tokens):

        # BS, seq_len -> BS, seq_len, dim 
        x = self.token_embedding(tokens)

        x += self.pos_embedding # literally add the position info compared to rope 

        return x

class CLIPLayer(nn.Module):
    
    def __init__(self, n_head, n_embd):
        super.__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)

        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):

        residue = x # input is BS, seq_len, dim 

        # attn

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        # FFN

        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # quick GeLU?

        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module):

    def __init__(self):

        self.embedding = CLIPEmbedding(49408, 768, 77) # vocab size, embedding size, and seq_len accounting for padding

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12) # n_head, embd_dim, 12 layers 
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor: # FloatTensor is normal, Longtensor because the tokenizer ids fit int64

        tokens = tokens.type(torch.long) # type conversion

        state = self.embedding(tokens) # BS, seq_len -> BS, seq_len, dim 

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state) # BS, seq_len, dim

        return output








