import torch
from torch import nn
from torch.nn import functional as F 

from decoder import VAE_attn_block, VAE_resid_block

class VAE_encoder(nn.Sequential):

    def __init__(self): # basically constantly reduce the total pixel size, but increase per pixel info 
        super().__init__(
            # (bs, Ch, H, W) -> (bs, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (bs, 128, H, W) -> (bs, 128, H, W)
            VAE_resid_block(128, 128),

            # (bs, 128, H, W) -> (bs, 128, H, W)
            VAE_resid_block(128, 128),

            # (bs, 128, H, W) -> (bs, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (bs, 128, H/2, W/2) -> (bs, 256, H/2, W/2)
            VAE_resid_block(128, 256),

            # (bs, 256, H/2, W/2) -> (bs, 256, H/2, W/2)
            VAE_resid_block(256, 256),

            # (bs, 256, H/2, W/2) -> (bs, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (bs, 256, H/4, W/4) -> (bs, 512, H/4, W/4)
            VAE_resid_block(256, 512),

            # (bs, 512, H/4, W/4) -> (bs, 512, H/4, W/4)
            VAE_resid_block(512, 512),

            # (bs, 512, H/4, W/4) -> (bs, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_resid_block(512, 512),

            VAE_resid_block(512, 512),

            # (bs, 512, H/8, W/8) -> (bs, 512, H/8, W/8)
            VAE_resid_block(512, 512),

            # (bs, 512, H/8, W/8) -> (bs, 512, H/8, W/8) 
            VAE_attn_block(512),

            # (bs, 512, H/8, W/8) -> (bs, 512, H/8, W/8)
            VAE_resid_block(512, 512),

            # (bs, 512, H/8, W/8) -> (bs, 512, H/8, W/8)
            nn.GroupNorm(32, 512),

            # (bs, 512, H/8, W/8) -> (bs, 512, H/8, W/8)
            nn.SiLU(),

            # known as the bottleneck of the model, 512->8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # 8->8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        # x: (bs, Ch, H, W), should be 512x512
        # noise: (bs, Out_Ch, H/8, W/8)

        for module in self:
            if getattr(module, "stride", None) == (2,2): # special stride padding 

                # (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0,1,0,1))
            
            x = module(x)

        # (bs, 8, H/8, W/8), turns into two tensors (bs, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)
        
        variance = log_variance.exp() # exponential to transform from log to regular

        stdev = variance.sqrt()

        # Z=N(0,1) -> N(mean, variance)=X 
        # X = mean + stdev * Z
        x = mean + stdev * noise 

        # scale the output by a constant 
        x *= 0.18215 # this is just given for some reason, I'm pretty sure someone even found a better version of this, check that out 

        return x








