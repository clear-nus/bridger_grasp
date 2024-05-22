import numpy as np
import torch
import os, os.path as osp

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from se3dif.losses.ddpm_loss import DDPMLoss, convert_h2x, convert_x2h
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class Grasp_VAE():
    def __init__(self, model, batch, device='cpu', dim=3):
        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4, 4]
        self.device = device
        self.batch = batch

    def sample(self, num_sample=10, x=None):
        with torch.no_grad():
            if x is None:
                latent_sample = torch.randn((num_sample, self.model.latent_size)).to(self.device)
                context = self.model(torch.zeros((num_sample, 4, 4)).float().to(self.device),
                                     torch.zeros((num_sample,)).float().to(self.device))
                x_hat = self.model.decoder(torch.cat([context, latent_sample], dim=-1))
                return convert_x2h(x_hat), None
            else:
                latent_sample = torch.randn((num_sample, self.model.latent_size)).to(self.device)
                context = self.model(torch.zeros((num_sample, 4, 4)).float().to(self.device),
                                     torch.zeros((num_sample,)).float().to(self.device))
                x_hat = self.model.decoder(torch.cat([context, latent_sample], dim=-1))

                post_latent_sample = self.model.encoder(torch.cat([context, x], dim=-1)).sample()
                x_hat_post = self.model.decoder(torch.cat([context, post_latent_sample], dim=-1))
                return convert_x2h(x_hat), convert_x2h(x_hat_post)


if __name__ == '__main__':
    import torch.nn as nn


    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            return convert_h2x(H), convert_h2x(H), convert_h2x(H)


    ## 2. Grasp_AnnealedLD
    generator = Grasp_VAE(model())
    H_initial = torch.ones([6, 4, 4])
    H = generator.sample(num_sample=10)
    print(H)
