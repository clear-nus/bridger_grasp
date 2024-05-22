import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 output_dim, layers: int, activation=nn.ELU):
        super().__init__()
        self._output_shape = (output_dim,)
        self._layers = layers
        self._hidden_size = hidden_dim
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dim
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self._min = 0.01
        self._max = 10.0

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, 2 * int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        shape_len = len(features.shape)
        if shape_len == 3:
            batch = features.shape[1]
            length = features.shape[0]
            features = features.reshape(-1, features.shape[-1])
        dist_inputs = self.model(features)
        reshaped_inputs_mean = dist_inputs[..., :np.prod(self._output_shape)]
        reshaped_inputs_std = dist_inputs[..., np.prod(self._output_shape):]

        reshaped_inputs_std = torch.clamp(self.soft_plus(reshaped_inputs_std), min=self._min, max=self._max)

        if shape_len == 3:
            reshaped_inputs_mean = reshaped_inputs_mean.reshape(length, batch, -1)
            reshaped_inputs_std = reshaped_inputs_std.reshape(length, batch, -1)
        return torch.distributions.independent.Independent(
            torch.distributions.Normal(reshaped_inputs_mean, reshaped_inputs_std), len(self._output_shape))