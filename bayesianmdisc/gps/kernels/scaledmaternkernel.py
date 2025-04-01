import gpytorch
import torch

from bayesianmdisc.gps.base import NamedParameters
from bayesianmdisc.gps.kernels.base import Kernel, KernelOutput
from bayesianmdisc.gps.utility import validate_parameters_size
from bayesianmdisc.customtypes import Device, Tensor


class ScaledMaternKernel(Kernel):
    def __init__(
        self, smoothness_param: float, input_dims: int, jitter: float, device: Device
    ) -> None:
        super().__init__(
            num_hyperparameters=1 + input_dims,
            jitter=jitter,
            device=device,
        )
        self._kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=smoothness_param, ard_num_dims=input_dims)
        ).to(device)
        self._lower_limit_output_scale = 0.0
        self._lower_limit_length_scales = 0.0

    def forward(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        return gpytorch.add_jitter(self._kernel(x_1, x_2), self._jitter)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        output_scale = parameters[0]
        length_scales = parameters[1:]
        self._set_output_scale(output_scale)
        self._set_length_scales(length_scales)

    def _set_output_scale(self, output_scale: Tensor) -> None:
        if output_scale >= self._lower_limit_output_scale:
            self._kernel.outputscale = output_scale.clone().to(self._device)

    def _set_length_scales(self, length_scales: Tensor) -> None:
        current_length_scales = self._kernel.base_kernel.lengthscale
        updated_length_scales = torch.where(
            length_scales >= self._lower_limit_length_scales,
            length_scales,
            current_length_scales,
        )
        self._kernel.base_kernel.lengthscale = updated_length_scales.clone().to(
            self._device
        )

    def get_named_parameters(self) -> NamedParameters:
        return {
            "output_scale": self._kernel.outputscale,
            "length_scale": self._kernel.base_kernel.lengthscale,
        }
