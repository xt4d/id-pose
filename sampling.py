import math

import numpy as np

from contextlib import nullcontext
from PIL import Image
from einops import rearrange

import torch
from torch import autocast

from ldm.models.diffusion.ddim import DDIMSampler

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision=='autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples,1,1)
            T = torch.tensor([x, math.sin(y), math.cos(y), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()\
                               .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def sample_images(
    model,
    input_im,
    x=0.,
    y=0.,
    z=0.,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    precision='fp32',
    h=256,
    w=256,
    ):

    sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w,\
                                  ddim_steps, n_samples, scale, ddim_eta, x, y, z)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims