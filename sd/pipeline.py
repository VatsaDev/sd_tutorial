import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np 
from tqdm import tqdm 

from ddpm import DDPMSampler

width = 512
height = 512 
latents_width = 512 // 8
latents_height = 512 // 8

def generate(prompt, 
             neg_prompt, # he used uncond_prompt, neg better 
             input_image = None, 
             strength = 0.8, 
             do_cfg = True, 
             cfg_scale = 7.5,
             sampler_name = "ddpm", 
             n_steps = 50,
             models = {}, 
             seed = None,
             device = None,
             idle_device = None,
             tokenizer = None
        ): 

    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError("strength is out of 0-1 range")

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)

        if seed is None:
            generate_seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg: # using negative prompt and cfg 

            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids # prompt to tokens
            cond_tokens = torch.Tensor(cond_tokens, dtype=torch.long, device=device) # now a tensor of bs, seq_len 
            cond_context = clip(cond_tokens) # bs, seq_len, dim (768) 

            neg_tokens = tokenizer.batch_encode_plus([neg_prompt], padding="max_length", max_length=77).input_ids
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
            neg_context = clip(neg_tokens)

            context = torch.cat([cond_context, neg_context]) # 2, seq_len, dim or 2, 77, 768
            
        else:

            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids 
            tokens = torch.Tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens) # 1, 77, 768 

        to_idle(clip) # move the clip off VRAM when its done

        if sampler_name = "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_steps)
        else:
            raise ValueError("no other sampler") # change this once we add the ZFP noise sampler, and also the shortcut model sampler 

        latents_shape = (1, 4, latents_height, latents_width)

        if input_image: 

            # image to image 

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((width, height))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.Tensor(input_image_tensor, dtype=torch.float32) # H, W, C (C=3)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0) # bs, H, W, C 
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2) # bs, C, H, W
            
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else: 

            # text to image random noise start

            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):

            # 1, 320

            time_embedding = get_time_embedding(timestep).to(device)

            # bs, 4, latent_height, latent_width
            model_input = latents

            if do_cfg:

                model_input = model_input.repeat(2,1,1,1) # doubles batch, 1 for conditional, 1 unconditional

            # predicted noise 
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg: # split here 

                output_cond, output_neg = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_neg) + output_neg

            latents = sampler.step(timestep, latents, model_output) # remove noise from the noise 

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)

        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1) # bs, C, H, W -> bs, H, W, C
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]

def rescale(x, old_range, new_range, clamp=False):

    old_min, old_max = old_range 
    new_min, new_max = new_range 

    x -= old_min 
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min 

    if clamp:
        x = x.clamp(new_min, new_max)

    return x

def get_time_embedding(timestep):

    freq = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160) # 160

    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] # 1, 160 

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # 1, 320
