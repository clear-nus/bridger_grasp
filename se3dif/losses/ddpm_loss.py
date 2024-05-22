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


def convert_h2x(h):
    # convert SE4 to R6, input must be torch.tensor
    H_th = SO3_R3(R=h[..., :3, :3], t=h[..., :3, -1])
    x = H_th.log_map()
    return x


def convert_x2h(x):
    # covert R6 to SE4, input must be torch.tensor
    h = SO3_R3().exp_map(x).to_matrix()
    return h


class DDPMLoss:
    def __init__(self):
        self.train_interval = 1000
        self.beta_max = 0.3
        self.ot_ode = False

        self.field = 'ddpm'

        self.ddpm_noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.train_interval,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def q_sample(self, step, x0, ot_ode=False):
        z = torch.randn(x0.shape, device=x0.device)

        x_t = self.ddpm_noise_scheduler.add_noise(
            x0, z, step)

        return x_t, z

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        ## From Homogeneous transformation to axis-angle ##
        H = model_input['x_ene_pos']
        if 'task_id' in model_input:
            task_id = model_input['task_id']

        n_grasps = H.shape[1]
        c = model_input['visual_context']
        model.set_latent(c, batch=n_grasps)

        H = H.reshape(-1, 4, 4)
        x0 = convert_h2x(H)

        # step = torch.randint(0, self.interval, (H.shape[0],)).to(H.device) / float(self.interval)
        step = torch.rand(H.shape[0]).to(H.device)

        xt, noise = self.q_sample((step * self.train_interval).long(), x0, self.ot_ode)

        if 'task_id' in model_input:
            noise_pred = model(convert_x2h(xt), step, task_id=task_id)
        else:
            noise_pred = model(convert_x2h(xt), step)

        loss = F.mse_loss(noise_pred, noise)
        info = {self.field: None}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
