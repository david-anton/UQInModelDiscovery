import gpytorch
import torch

from bayesianmdisc.gps.base import NamedParameters
from bayesianmdisc.gps.means.base import MeanOutput, NonZeroMean
from bayesianmdisc.gps.utility import validate_parameters_size
from bayesianmdisc.customtypes import Device, Tensor


class LinearMean(NonZeroMean):
    def __init__(self, input_dim: int, device: Device) -> None:
        super().__init__(
            num_hyperparameters=input_dim,
            device=device,
        )
        self._mean = gpytorch.means.LinearMean(input_dim, bias=False).to(device)

    def forward(self, x: Tensor) -> MeanOutput:
        return self._mean(x)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        weights_mean = parameters.clone().reshape((-1, 1))
        self._mean.weights = torch.nn.Parameter(weights_mean).to(self._device)

    def get_named_parameters(self) -> NamedParameters:
        return {
            "weights_mean": self._mean.weights,
        }
