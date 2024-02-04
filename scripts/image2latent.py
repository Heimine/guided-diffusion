# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import argparse
import pickle
import numpy as np
import torch
import PIL.Image
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

#----------------------------------------------------------------------------

def image_2_latent(
    args, images,
    seed=0, gridw=8, gridh=8,
    num_steps=18
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Pick latents and labels.
    print(f'Embedding {batch_size} images...')

    sample_fn = diffusion.ddim_reverse_sample
    x_next = images
    for t in tqdm.tqdm(range(num_steps)):
        result = sample_fn(model, x_next, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,eta=0.0)
        x_next = result["sample"]

    return x_next

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

def latent_2_image(
    args, dest_path, latents = None,
    seed=0, gridw=8, gridh=8, img_resolution=256
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')

    model_kwargs = {}
    sample_fn = diffusion.ddim_sample_loop
    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        noise=latents, 
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    # sample = sample.contiguous()

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = sample
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * img_resolution, gridw * img_resolution, 3)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')
    return sample
#----------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()
    num_step = 1000
    images = latent_2_image(args, 'imagenet-64x64.png')
    print(images.shape)
    latent = image_2_latent(args, images, num_steps=num_step)
    print(latent.max())
    images = latent_2_image(args, 'imagenet-64x64-recon.png', latents = latent)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------