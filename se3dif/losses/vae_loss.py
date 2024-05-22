# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import numpy as np
import torch
from theseus import SO3
from se3dif.utils import SO3_R3
import theseus as th
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from functools import partial
import torch.nn.functional as F


def kl_divergence_normal(mean1, mean2, std1, std2):
    # mean1, std1: post dist
    # mean2 std2: prior dist
    kl_loss = ((std2 + 1e-9).log() - (std1 + 1e-9).log()
               + (std1.pow(2) + (mean2 - mean1).pow(2))
               / (2 * std2.pow(2) + 1e-9) - 0.5).sum(-1).mean()
    return kl_loss

def convert_h2x(h):
    # convert SE4 to R6, input must be torch.tensor
    H_th = SO3_R3(R=h[..., :3, :3], t=h[..., :3, -1])
    x = H_th.log_map()
    return x


def convert_x2h(x):
    # covert R6 to SE4, input must be torch.tensor
    h = SO3_R3().exp_map(x).to_matrix()
    return h


class VAELoss:
    def __init__(self):
        self.field = 'vae'
        self.anneal_factor = 0.0

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        ## From Homogeneous transformation to axis-angle ##
        H = model_input['x_ene_pos']

        n_grasps = H.shape[1]
        c = model_input['visual_context']
        model.set_latent(c, batch=n_grasps)

        H = H.reshape(-1, 4, 4)
        x = convert_h2x(H)

        context = model(torch.zeros_like(H).float().to(H.device), torch.zeros(H.shape[0]).float().to(H.device))
        latent_post_dist = model.encoder(torch.cat([context, x], dim=-1))
        latent_post_rsample = latent_post_dist.rsample()
        latent_post_mean = latent_post_dist.mean
        latent_post_std = latent_post_dist.stddev

        latent_prior_mean = torch.zeros_like(latent_post_mean).float().to(H.device)
        latent_prior_std = torch.ones_like(latent_post_std).float().to(H.device)

        # reconstruction loss
        x_rec = model.decoder(torch.cat([context, latent_post_rsample], dim=-1))
        rec_loss = F.mse_loss(x_rec, x)
        kl_loss = self.anneal_factor * kl_divergence_normal(latent_post_mean, latent_prior_mean, latent_post_std, latent_prior_std)

        self.anneal_factor += 0.0001
        self.anneal_factor = 0.1 if self.anneal_factor > 0.1 else self.anneal_factor
        # print(self.anneal_factor)
        info = {self.field: None}
        loss_dict = {"rec loss": rec_loss, "kl loss": kl_loss}
        return loss_dict, info
