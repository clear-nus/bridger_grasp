import numpy as np
import torch
import os, os.path as osp

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
from se3dif.losses.si_loss import unsqueeze_xdim, StochasticInterpolantsLoss, convert_h2x, convert_x2h


class Grasp_SI():
    def __init__(self, model, device='cpu', batch=10, dim=3, k_steps=1,
                 T=50, T_fit=5, deterministic=False):

        self.model = model
        self.device = device
        self.dim = dim
        self.shape = [4, 4]
        self.batch = batch

        self.si = StochasticInterpolantsLoss()

        self.interpolant_type = self.si.interpolant_type
        self.gamma_type = self.si.gamma_type
        self.epsilon_type = self.si.epsilon_type
        self.interval = T

        self.d = self.si.d

        self.t_min = self.si.t_min
        self.gamma_inv_max = self.si.gamma_inv_max

    def sde_bs(self, x_initial, task_id, score_weight=1.0, direction='forward'):
        # Number of steps and samples
        n_steps = self.interval
        delta_t = 1.0 / n_steps
        n_samples = x_initial.shape[0]

        # Create a tensor to hold the samples at each time step
        x_values = [[]] * (n_steps + 1)
        x_values[0] = x_initial

        b_values = []
        s_values = []

        # Simulate the SDE
        for t in range(1, n_steps + 1):
            current_x = x_values[t - 1]

            # Create a tensor of shape (n_samples, 1) filled with the current time value
            t_tensor = torch.full((n_samples,), t / n_steps).float().to(x_initial.device)
            t_tensor = torch.clip(t_tensor, self.t_min, 1.0 - self.t_min)

            if direction == 'forward':
                _, s_value, b_value = self.model(convert_x2h(current_x), t_tensor)
                gamma_inv = self.si.gamma_inv(t_tensor)
            elif direction == 'backward':
                _, s_value, b_value = self.model(convert_x2h(current_x), 1.0 - t_tensor)
                gamma_inv = self.si.gamma_inv(1.0 - t_tensor)
            else:
                raise NotImplementedError

            s_value = s_value.detach()
            b_value = b_value.detach()

            b_values += [b_value.cpu().numpy()]
            s_values += [s_value.cpu().numpy()]

            batch, *xdim = s_value.shape
            gamma_inv = unsqueeze_xdim(gamma_inv, xdim)
            s_value = s_value * gamma_inv

            # Generate the Wiener process increment
            dW = self.d * torch.randn_like(current_x).float().to(x_initial.device)

            if direction == 'forward':
                noise_scale = delta_t * torch.sqrt(2 * self.si.epsilon(t_tensor[0]))
                score_epsilon = score_weight * self.si.epsilon(t_tensor[0])
                new_x = current_x + (b_value + score_epsilon * s_value) * delta_t
            elif direction == 'backward':
                noise_scale = delta_t * torch.sqrt(2 * self.si.epsilon(1.0 - t_tensor[0]))
                score_epsilon = score_weight * self.si.epsilon(1.0 - t_tensor[0])
                new_x = current_x - (b_value - score_epsilon * s_value) * delta_t
            else:
                raise NotImplementedError
            new_x += noise_scale * dW
            x_values[t] = new_x

        b_values = np.stack(b_values, axis=0)
        s_values = np.stack(s_values, axis=0)

        np.savez('/home/yongli/project/grasp_diff/linear_heuristic.npz',
                 b_values=b_values,
                 s_values=s_values)

        return x_values[-1], x_values

    def sde_vs(self, x_initial, task_id=None, score_weight=1.0, direction='forward'):
        n_steps = self.interval
        delta_t = 1.0 / n_steps
        n_samples = x_initial.shape[0]

        # Create a tensor to hold the samples at each time step
        x_values = [[]] * (n_steps + 1)
        x_values[0] = x_initial

        decays = torch.linspace(score_weight, 1.0, n_steps + 1).float().to(x_initial.device)
        # Simulate the SDE
        for t in range(1, n_steps + 1):
            current_x = x_values[t - 1]

            # Create a tensor of shape (n_samples, 1) filled with the current time value
            t_tensor = torch.full((n_samples,), t / n_steps).float().to(x_initial.device)
            t_tensor = torch.clip(t_tensor, self.t_min, 1.0 - self.t_min)

            if direction == 'forward':
                gamma_t, dot_gamma_t = self.si.gamma(t_tensor), self.si.gamma_der(t_tensor)
                if task_id is None:
                    v_value, s_value, _ = self.model(convert_x2h(current_x), t_tensor)
                else:
                    v_value, s_value, _ = self.model(convert_x2h(current_x), t_tensor, task_id)
                gamma_inv = self.si.gamma_inv(t_tensor)
            elif direction == 'backward':
                gamma_t, dot_gamma_t = self.si.gamma(1.0 - t_tensor), self.si.gamma_der(1.0 - t_tensor)
                if task_id is None:
                    v_value, s_value, _ = self.model(convert_x2h(current_x), 1.0 - t_tensor)
                else:
                    v_value, s_value, _ = self.model(convert_x2h(current_x), 1.0 - t_tensor, task_id)
                gamma_inv = self.si.gamma_inv(1.0 - t_tensor)
            else:
                raise NotImplementedError

            batch, *xdim = s_value.shape
            gamma_inv = unsqueeze_xdim(gamma_inv, xdim)
            s_value = s_value * gamma_inv

            dot_gamma_gamma_t = dot_gamma_t.float().to(x_initial.device) * gamma_t.float().to(x_initial.device)
            dot_gamma_gamma_t = unsqueeze_xdim(dot_gamma_gamma_t, xdim)
            b_value = v_value - dot_gamma_gamma_t * s_value * torch.sqrt(2 * self.si.epsilon(t_tensor[0]))

            # Generate the Wiener process increment
            dW = self.d * torch.randn_like(current_x).float().to(x_initial.device)
            if t > 0.7 * n_steps:
                dW = dW * 0.01

            if direction == 'forward':
                noise_scale = delta_t * torch.sqrt(2 * self.si.epsilon(t_tensor[0]))
                score_epsilon = score_weight * self.si.epsilon(t_tensor[0])
                new_x = current_x + (b_value + score_epsilon * decays[t] * s_value) * delta_t
            elif direction == 'backward':
                noise_scale = delta_t * torch.sqrt(2 * self.si.epsilon(1.0 - t_tensor[0]))
                score_epsilon = score_weight * self.si.epsilon(1.0 - t_tensor[0])
                new_x = current_x - (b_value - score_epsilon * s_value) * delta_t
            else:
                raise NotImplementedError
            new_x += noise_scale * dW # * decays[t]
            x_values[t] = new_x

        return x_values[-1], x_values

    def sample(self, H_initial, task_id=None, sde_type='vs', save_path=False):
        """

        :param x_inital: (batch, feature)
        :param sde_type: vs, bs
        :param save_path:
        :return:
        """
        with torch.no_grad():
            x_initial = convert_h2x(H_initial)
            if sde_type == 'vs':
                x_target, x_target_traj = self.sde_vs(x_initial=x_initial, task_id=task_id, score_weight=5.0, direction='forward')
            elif sde_type == 'bs':
                x_target, x_target_traj = self.sde_bs(x_initial=x_initial, task_id=task_id, score_weight=5.0, direction='forward')
            else:
                raise NotImplementedError

            H_target = convert_x2h(x_target)
            if save_path:
                H_target_traj = []
                for x_target_item in x_target_traj:
                    H_target_traj += [convert_x2h(x_target_item)]
                return H_target, H_target_traj
            else:
                return H_target


if __name__ == '__main__':
    import torch.nn as nn

    class model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, H, k):
            return convert_h2x(H), convert_h2x(H), convert_h2x(H)

    ## 2. Grasp_AnnealedLD
    generator = Grasp_SI(model(), T=100)
    H_initial = torch.ones([6, 4, 4])
    H = generator.sample(H_initial=H_initial)
    print(H)




