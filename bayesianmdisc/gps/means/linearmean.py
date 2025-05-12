import gpytorch
import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.gps.base import NamedParameters
from bayesianmdisc.gps.means.base import MeanOutput, NonZeroMean
from bayesianmdisc.gps.utility import validate_parameters_size


class LinearMean(NonZeroMean):
    def __init__(self, input_dim: int, device: Device) -> None:
        self._num_weights = input_dim
        self._num_bias = 1
        num_hyperparameters = self._num_weights + self._num_bias
        super().__init__(
            num_hyperparameters=num_hyperparameters,
            device=device,
        )
        self._mean = gpytorch.means.LinearMean(input_dim, bias=True).to(device)

    def forward(self, x: Tensor) -> MeanOutput:
        return self._mean(x)

    def set_parameters(self, parameters: Tensor) -> None:
        weights = parameters[: self._num_weights].clone().reshape((-1, 1))
        validate_parameters_size(weights, torch.Size([self._num_weights, 1]))
        bias = parameters[-1].clone()
        validate_parameters_size(bias, torch.Size([]))
        self._mean.weights = torch.nn.Parameter(weights).to(self._device)
        self._mean.bias = torch.nn.Parameter(bias).to(self._device)

    def get_named_parameters(self) -> NamedParameters:
        return {
            "weights_mean": self._mean.weights,
            "bias_mean": self._mean.bias,
        }
