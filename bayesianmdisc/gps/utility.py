from typing import TypeAlias

import gpytorch
import torch

from bayesianmdisc.errors import GPError
from bayesianmdisc.types import Tensor, TensorSize

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]


def validate_parameters_size(
    parameters: Tensor, valid_parameter_size: int | TensorSize
) -> None:
    parameters_size = parameters.size()
    if isinstance(valid_parameter_size, int):
        valid_parameter_size = torch.Size([valid_parameter_size])
    if parameters_size != valid_parameter_size:
        raise GPError(f"Parameter tensor has unvalid size {parameters_size}")
