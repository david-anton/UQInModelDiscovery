import pytest
import torch

from bayesianmdisc.errors import GPError
from bayesianmdisc.gps.utility import validate_parameters_size
from bayesianmdisc.types import Tensor, TensorSize


@pytest.mark.parametrize(
    ("parameters", "valid_parameter_size"),
    [
        (torch.tensor([1]), 1),
        (torch.tensor([1]), torch.Size([1])),
        (torch.tensor([1, 1]), torch.Size([2])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([2, 2])),
    ],
)
def test_validate_parameters_size_for_valid_parameters(
    parameters: Tensor, valid_parameter_size: int | TensorSize
) -> None:
    sut = validate_parameters_size
    try:
        sut(parameters=parameters, valid_parameter_size=valid_parameter_size)
    except GPError:
        assert False


@pytest.mark.parametrize(
    ("parameters", "valid_parameter_size"),
    [
        (torch.tensor([1]), 2),
        (torch.tensor([1]), torch.Size([2])),
        (torch.tensor([1, 1]), torch.Size([1])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([1, 1])),
    ],
)
def test_validate_parameters_size_for_unvalid_parameters(
    parameters: Tensor, valid_parameter_size: int | TensorSize
) -> None:
    sut = validate_parameters_size
    with pytest.raises(GPError):
        sut(parameters=parameters, valid_parameter_size=valid_parameter_size)
