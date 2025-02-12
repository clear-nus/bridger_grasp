# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import numpy as np
import torch
from se3dif.utils import SO3_R3

def indicator_function(condition):
    # Create a tensor of zeros with the same shape as the condition
    result = torch.zeros_like(condition, dtype=torch.float32)

    # Set the elements to 1 where the condition is satisfied
    result[condition] = 1.0

    return result


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def convert_h2x(h):
    # convert SE4 to R6, input must be torch.tensor
    H_th = SO3_R3(R=h[..., :3, :3], t=h[..., :3, -1])
    x = H_th.log_map()
    return x


def convert_x2h(x):
    # covert R6 to SE4, input must be torch.tensor
    h = SO3_R3().exp_map(x).to_matrix()
    return h


class StochasticInterpolantsLoss:
    def __init__(self, ):

        self.interpolant_type = 'power3'
        self.gamma_type = '(2t(t-1))^0.5'
        self.epsilon_type = '1-t'
        self.interval = 100

        self.d = 0.3

        self.field = 'si'

        self.t_min = 0.001
        self.gamma_inv_max = 200.0

    def epsilon(self, t):
        if self.epsilon_type == 't(t-1)':
            return t * (1 - t)
        elif self.epsilon_type == '1-t':
            return (1 - t) * 1.0
        elif self.epsilon_type == '1-sqrt(t)':
            return 1 - torch.sqrt(t)
        elif self.epsilon_type == '1-t^2':
            return 1 - torch.pow(t, 2)
        elif self.epsilon_type == '0':
            return t * 0.0
        else:
            raise NotImplementedError

    def gamma(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return 1.4142 * torch.sqrt(t * (1 - t))
        elif self.gamma_type == '2^0.5*t(t-1)':
            return 1.4142 * t * (1 - t)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return 1.4142 * torch.pow((1 - t), 2.0) * torch.sqrt(t)
        else:
            raise NotImplementedError

    def gamma_der(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return (1 - 2 * t) / torch.sqrt(2 * (t - torch.pow(t, 2)) + 1e-4)
        if self.gamma_type == 't(t-1)':
            return 1.4142 * (1 - 2 * t)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return 1.4142 * (2 * (t - 1) * torch.sqrt(t) + torch.pow((1 - t), 2.0) / (2.0 * torch.sqrt(t + 1e-4)))
        else:
            raise NotImplementedError

    def gamma_inv(self, t):
        if self.gamma_type == '(2t(t-1))^0.5':
            return torch.clamp(1 / (1.4142 * torch.sqrt(t * (1 - t) + 1e-4)), 0.0, self.gamma_inv_max)
        elif self.gamma_type == 't(t-1)':
            return torch.clamp(1 / (1.4142 * t * (1 - t) + 1e-4), 0.0, self.gamma_inv_max)
        elif self.gamma_type == '(1-t)^2(2t)^0.5':
            return torch.clamp(1 / (1.4142 * torch.pow((1 - t), 2.0) * torch.sqrt(t) + 1e-4), 0.0, self.gamma_inv_max)
        else:
            raise NotImplementedError

    def velocity_loss(self, v, t, x_0, x_1):
        if self.interpolant_type == 'linear':
            partial_t = (x_1 - x_0)
        elif self.interpolant_type == 'power3':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(1 - t_reshape, 2) * (x_1 - x_0)
        elif self.interpolant_type == 'power4':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(1 - t_reshape, 3) * (x_1 - x_0)
        elif self.interpolant_type == 'reverse_power3':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(t_reshape, 2) * (x_1 - x_0)
        elif self.interpolant_type == 'reverse_power4':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(t_reshape, 3) * (x_1 - x_0)
        elif self.interpolant_type == 'gaussian_encode_decode':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape <= 0.5) * x_0
            partial_t += -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape > 0.5) * x_1
        elif self.interpolant_type == 'reverse_linear':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * indicator_function(t_reshape <= 0.5) * x_0
            partial_t += 2 * indicator_function(t_reshape <= 0.5) * x_1
        else:
            raise NotImplementedError
        v_reshape = v  # .flatten(-2)
        partial_t_reshape = partial_t  # .flatten(-2)

        loss = 0.5 * torch.norm(v_reshape, dim=-1) ** 2 - torch.sum(partial_t_reshape * v_reshape, dim=-1)
        return torch.mean(loss)

    # Do the same for other loss functions
    def score_loss(self, s, t, z):
        gamma_inv = self.gamma_inv(t)

        s_reshape = s  # .flatten(-2)
        z_reshape = z  # .flatten(-2)
        # loss = gamma_inv * gamma_inv * (0.5 * torch.norm(s_reshape, dim=-1) ** 2 + torch.sum(z_reshape * s_reshape, dim=-1))
        loss = (0.5 * torch.norm(s_reshape, dim=-1) ** 2 + torch.sum(z_reshape * s_reshape, dim=-1))
        return torch.mean(loss)

    def b_loss(self, b, t, x_0, x_1, z):
        if self.interpolant_type == 'linear':
            partial_t = (x_1 - x_0)
        elif self.interpolant_type == 'reverse_power3':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(t_reshape, 2) * (x_1 - x_0)
        elif self.interpolant_type == 'reverse_power4':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(t_reshape, 3) * (x_1 - x_0)
        elif self.interpolant_type == 'power3':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 3 * torch.pow(1 - t_reshape, 2) * (x_1 - x_0)
        elif self.interpolant_type == 'power4':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = 4 * torch.pow(1 - t_reshape, 3) * (x_1 - x_0)
        elif self.interpolant_type == 'gaussian_encode_decode':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape <= 0.5) * x_0
            partial_t += -2 * torch.pi * torch.cos(torch.pi * t_reshape) * torch.sin(torch.pi * t_reshape) * indicator_function(t_reshape > 0.5) * x_1
        elif self.interpolant_type == 'reverse_linear':
            batch, *xdim = x_1.shape
            t_reshape = unsqueeze_xdim(t, xdim)
            partial_t = -2 * indicator_function(t_reshape <= 0.5) * x_0
            partial_t += 2 * indicator_function(t_reshape <= 0.5) * x_1
        else:
            raise NotImplementedError

        gamma_der = self.gamma_der(t)
        b_reshape = b  # .flatten(-2)
        partial_t_reshape = partial_t  # .flatten(-2)

        batch, *xdim = b_reshape.shape
        gamma_der_reshape = unsqueeze_xdim(gamma_der, xdim)

        z_reshape = z  # .flatten(-2)

        loss = 0.5 * torch.norm(b_reshape, dim=-1) ** 2 - torch.sum((partial_t_reshape + gamma_der_reshape * z_reshape) * b_reshape, dim=-1)
        # loss = torch.square(b_reshape - (partial_t_reshape + gamma_der_reshape * z_reshape)).sum(-1)
        return torch.mean(loss)

    def q_sample(self, t, x0, x1):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """
        batch, *xdim = x0.shape
        t_batch = unsqueeze_xdim(t, xdim)
        t_batch = torch.clip(t_batch, self.t_min, 1.0 - self.t_min)

        gamma = self.gamma(t_batch)
        if self.interpolant_type == 'linear':
            z = self.d * torch.randn_like(x0).float().to(x0.device)

            xt = (1 - t_batch) * x0 + t_batch * x1 + gamma * z

        elif self.interpolant_type == 'reverse_power3':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = 1 - torch.pow(t_batch, 3)
            w_x1 = torch.pow(t_batch, 3)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'reverse_power4':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = 1 - torch.pow(t_batch, 4)
            w_x1 = torch.pow(t_batch, 4)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'power3':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow((1 - t_batch), 3)
            w_x1 = 1 - torch.pow((1 - t_batch), 3)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'power4':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow((1 - t_batch), 4)
            w_x1 = 1 - torch.pow((1 - t_batch), 4)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'gaussian_encode_decode':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = torch.pow(torch.cos(t_batch * np.pi), 2) * indicator_function(t_batch <= 0.5)
            w_x1 = torch.pow(torch.cos(t_batch * np.pi), 2) * indicator_function(t_batch > 0.5)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        elif self.interpolant_type == 'reverse_linear':
            z = self.d * torch.randn_like(x0).float().to(x0.device)
            w_x0 = (1 - 2 * t_batch) * indicator_function(t_batch <= 0.5)
            w_x1 = 1 - (1 - 2 * t_batch) * indicator_function(t_batch <= 0.5)

            xt = w_x0 * x0 + w_x1 * x1 + gamma * z
        else:
            raise NotImplementedError

        return xt.detach(), z

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

        # step = torch.randint(0, self.interval, (H.shape[0],)).to(H.device) / float(self.interval)
        step = torch.rand(H.shape[0]).to(H.device)

        source = x0
        target = x1
        xt, noise = self.q_sample(step, source, target)

        # xt = torch.clip(xt, -1, 1)
        # noise = torch.clip(noise, -1, 1)

        step = torch.clip(step, self.t_min, 1 - self.t_min)
        if 'task_id' in model_input:
            v, s, b = model(convert_x2h(xt), step, task_id)
        else:
            v, s, b = model(convert_x2h(xt), step)
        v_loss = self.velocity_loss(v=v, t=step, x_0=source, x_1=target)
        s_loss = self.score_loss(s=s, t=step, z=noise)
        b_loss = self.b_loss(b=b, t=step, x_0=source, x_1=target, z=noise)

        loss = v_loss + s_loss + b_loss
        info = {self.field: None}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
