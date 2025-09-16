import torch
from torch import nn
from torch.nn import functional as F

from ddpm import DDPMSampler
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import TimeEmbedding, UNET_ResidualBlock, UNET_AttentionBlock, Upsample, SwitchSequential, UNET_OutputLayer

# currently 12M params

class MicroUNET(nn.Module):
    """
    An even smaller U-Net architecture (~6M params).
    This version follows the same symmetric design as TinyUNET but with
    fewer channels to reduce the parameter count by about half.
    """
    def __init__(self):
        super().__init__()
        
        # All residual blocks accept the 1280-dim output of the TimeEmbedding module.
        time_emb_dim = 1280

        # --- Encoder ---
        # Channel progression: 4 -> 96 -> 192 -> 256
        self.enc_input = SwitchSequential(nn.Conv2d(4, 96, kernel_size=3, padding=1))
        self.enc_res1 = SwitchSequential(UNET_ResidualBlock(96, 96, n_time=time_emb_dim))
        self.enc_down1 = SwitchSequential(nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)) # -> H/2
        
        self.enc_res2 = SwitchSequential(UNET_ResidualBlock(96, 192, n_time=time_emb_dim))
        self.enc_down2 = SwitchSequential(nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)) # -> H/4
        
        self.enc_res3 = SwitchSequential(UNET_ResidualBlock(192, 256, n_time=time_emb_dim))
        
        # --- Bottleneck ---
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(256, 256, n_time=time_emb_dim),
            UNET_AttentionBlock(8, 256), # 8 heads, 256 channels
            UNET_ResidualBlock(256, 256, n_time=time_emb_dim),
        )

        # --- Decoder ---
        # Channel progression: (256+256) -> 192 -> (192+192) -> 96 -> (96+96) -> 96
        self.dec_res1 = SwitchSequential(UNET_ResidualBlock(256 + 256, 192, n_time=time_emb_dim))
        self.dec_up1 = SwitchSequential(Upsample(192)) # -> H/2
        
        self.dec_res2 = SwitchSequential(UNET_ResidualBlock(192 + 192, 96, n_time=time_emb_dim))
        self.dec_up2 = SwitchSequential(Upsample(96)) # -> H
        
        self.dec_res3 = SwitchSequential(UNET_ResidualBlock(96 + 96, 96, n_time=time_emb_dim))

    def forward(self, x, context, time):
        # --- Encoder ---
        x = self.enc_input(x, context, time)
        skip1 = self.enc_res1(x, context, time)     # Resolution: H
        
        x = self.enc_down1(skip1, context, time)
        skip2 = self.enc_res2(x, context, time)     # Resolution: H/2
        
        x = self.enc_down2(skip2, context, time)
        skip3 = self.enc_res3(x, context, time)     # Resolution: H/4
        
        # --- Bottleneck ---
        x = self.bottleneck(skip3, context, time)
        
        # --- Decoder ---
        x = torch.cat((x, skip3), dim=1)
        x = self.dec_res1(x, context, time)
        x = self.dec_up1(x, context, time)
        
        x = torch.cat((x, skip2), dim=1)
        x = self.dec_res2(x, context, time)
        x = self.dec_up2(x, context, time)

        x = torch.cat((x, skip1), dim=1)
        x = self.dec_res3(x, context, time)
        
        return x


class DiffusionModel(nn.Module):
    """
    The main model wrapper that combines the time embedding and the U-Net.
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = MicroUNET()
        # The final layer must match the output channels of the last decoder block (128)
        self.final = UNET_OutputLayer(96, 4)

    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output

# Helper function to generate the initial time embedding for a forward pass
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionModel().to(device)

    # Print Parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.2f}M trainable parameters.")

    # Sanity Check Forward Pass
    try:
        batch_size = 2
        latent_h, latent_w = 32, 32 # Latent dimensions must be divisible by 4
        seq_len, d_context = 77, 768

        latents = torch.randn(batch_size, 4, latent_h, latent_w, device=device)
        context = torch.randn(batch_size, seq_len, d_context, device=device)
        time = get_time_embedding(50).to(device)

        output = model(latents, context, time)
        print(f"Sanity check passed! Output shape: {output.shape}")
        assert output.shape == latents.shape
    except Exception as e:
        print(f"Sanity check failed with error: {e}")


    # training loop
    print("\nTraining script skeleton ready. Implement your training loop below.")
    
    # 1. Instantiate VAE, pre-trained and frozen.
    # 2. Instantiate the DDPM Sampler
    # 3. Setup Optimizer
    # 4. Create your DataLoader
    # 5. The Actual Loop
