from clip import CLIP 
from encoder import VAE_encoder
from decoder import VAE_decoder
from diffusion import Diffusion 

import model_converter

def preload_model_from_standard_weights(ckpt_path, device):

    state_dict = model_converter.load_from_standard_weights(ckpt_path, device) # uses the mapping script

    encoder = VAE_encoder.to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_decoder.to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion.to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP.to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
