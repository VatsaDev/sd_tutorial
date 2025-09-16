import torch
from torch import nn
from torch.nn import functional as F 

from attention import selfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels):
        super.__init__()

        self.groupnorm = nn.GroupNorm(32, channels) # normalizes in groups, works well because images usually have closer pixels mean more than farther pixels
        self.attention = selfAttention(1, channels)

    def forward(self, x):
        
        # (bs, channels, H, W)

        residue = x

        n, c, h, w = x.shape # independant vars for all dims of the tensor

        x = x.view(n, c, h*w)   # (bs, ch, h*w)
        x = x.transpose(-1, -2) # (bs, h*w, features)

        x = self.attention(x)

        x = x.transpose(-1, -2) # (bs, features, h*w)
        x = x.view(n, c, h, w)  # (bs, ch, h, w)

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super.__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # skip connections, this class is a wrapper that adapts the channels to the output
        
        if in_channels==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        def forward(self, x):

            # (bs, in_channels, H, W)

            residue = x

            x = self.groupnorm_1(x)
            x = F.silu(x)
            x = self.conv_1(x)

            x = self.groupnorm_2(x)
            x = F.silu(x)
            x = self.conv_2(x)

            return x + self.residual_layer(x) # just wraps the channels, thats actually really convenient, use it places 

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super.__init__(

            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512), # BS, 512, h/8, w/8

            nn.Upsample(scale_factor=2), # BS, 512, h/4, w/4

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.Upsample(scale_factor=2), # BS, 512, h/2, w/2

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            nn.Upsample(scale_factor=2), # BS, 256, H, W 

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128), # 128 features, 32 groups 

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1) # BS, 3, H, W
        )

    def forward(x):

        x /= 0.18215 # undo prev scaling 

        for module in self:
            x = module(x)

        return x


