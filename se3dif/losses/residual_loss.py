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


class ResidualLoss:
    def __init__(self):
        self.residual_weight = 1.0

        self.field = 'residual'

    def q_sample(self, x0, x1):
        return x1-x0

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        ## From Homogeneous transformation to axis-angle ##
        H = model_input['x_ene_pos']
        H_prior = model_input['x_ene_pos_prior']
        if 'task_id' in model_input:
            task_id = model_input['task_id']

        n_grasps = H.shape[1]
        c = model_input['visual_context']
        model.set_latent(c, batch=n_grasps)

        H = H.reshape(-1, 4, 4)
        H_prior = H_prior.reshape(-1, 4, 4)
        x1 = convert_h2x(H)
        x0 = convert_h2x(H_prior)

        residual = self.q_sample(x0=x0, x1=x1)
        step = torch.zeros(H.shape[0]).to(H.device)

        if 'task_id' in model_input:
            residual_pred = model(convert_x2h(x0), step, task_id=task_id)
        else:
            residual_pred = model(convert_x2h(x0), step)

        loss = F.mse_loss(residual_pred, residual)
        info = {self.field: None}
        loss_dict = {"Residual loss": loss}
        return loss_dict, info
