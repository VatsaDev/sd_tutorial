import torch
import torch.nn as nn
from torch.nn import functional as F 

from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):

        x = self.linear_1(x) # 1, 320
        x = F.silu(x)
        x = self.linear_2(x) # 1, 1280

        return x

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_time):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):

        # feature   -> bs, in_channels, H, W
        # time_embd -> 1, 1280 

        residue = feature 

        feature = self.groupnorm(feature)
        feature = F.silu(feature)
        feature = self.conv(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # figure out shape of this one
        merged = self.groupnorm_merged(merged) 
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()

        channels = n_head*n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gegelu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_gegelu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):

        # x: bs, channels, H, W 
        # ctx: bs, seq_len, dim (another name for the d_context 768)

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape 

        x = x.view((n, c, h*w)) # bs, C, H, W -> bs, C, H*W

        x = x.transpose(-1, -2) # bs, C, H*W -> bs, H*w, C (switched last and second last)

        # normalizationa and attn 

        # first add self attn 

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)

        x += residue_short 

        # now new residual for cross attn 

        residue_short = x

        x = self.layernorm_2(x)
        self.attention_2(x, context)

        x += residue_short 

        # gelu stuff for attn here 

        residue_short = x

        x = self.layernorm_3(x)

        x, gate = self.linear_gegelu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_gegelu_2(x)
        x += residue_short

        x = x.transpose(-1, -2) # bs, H*W, C -> bs, C, H*W 
        x = x.view((n, c, h, w)) # now bs, C, H, W

        return self.conv_output(x) + residue_long     

class UpSample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):

        # bs, channels, H, W -> bs, channels, H*2, W*2
        x = F.interpolate(x, scale_factor=2, mode="nearest") # if I remember correctly, nearest is one of the worst, the lucas beyer VLM had the better options
        
        return self.conv(x)

class SwitchSequential(nn.Sequential): # switch case cover for all the tensors that get put in, simplfier

    def forward(self, x, context, time):

        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([

            # BS, 4, H/8, W/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), # like an nn.sequential, but you can do things diff according to params?
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # BS, 320, H/16, W/16
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)), # atten block is going from heads to embd size

            # BS, 640, H/32, W/32
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # BS, 1280, H/64, W/64
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # BS, 1280, H/64, W/64
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), # while the bottleneck output is 1280, the skip connect is also the same size, doubling the input
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)), # custom UpSample

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)), # generally keep downsizing features, upsampling size
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)), # somewhere we downsampled to a 1920? unsure

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)), # I'm not really sure why these shapes are what they are, some 40, 80 etc, its odd
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), # we dont upsample here, give it to the forward to rebuild
        ])

class UNET_OutputLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        # bs, 320, H, W -> bs, 4, H, W 
        
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x

class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):

        # latent, BS, 4, H/8, W/8
        # context, BS, seq_len, dim
        # time, (1, 320) 

        time = self.time_embedding(time) # t, 1, 1280

        output = self.unet(latent, context, time) # Bs, 4, h/8, h/8 -> Bs, 320, h/8, w/8

        output = self.final(output) # Bs, 320, h/8, h/8 -> Bs, 4, h/8, h/8

        return output
