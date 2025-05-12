import pytest
import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.gps.means.linearmean import LinearMean

device = torch.device("cpu")
weights_mean = 2.0
bias_mean = 1.0


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([1.0]), torch.tensor([3.0])),
        (torch.tensor([0.0]), torch.tensor([1.0])),
        (torch.tensor([-1.0]), torch.tensor([-1.0])),
        (torch.tensor([1.0, 0.0, -1.0]), torch.tensor([3.0, 1.0, -1.0])),
        (torch.tensor([[1.0], [0.0], [-1.0]]), torch.tensor([3.0, 1.0, -1.0])),
    ],
)
def test_linear_mean_one_dimensional_input(inputs: Tensor, expected: Tensor) -> None:
    input_dim = 1
    sut = LinearMean(input_dim, device=device)
    parameters = torch.concat(
        (
            torch.full((input_dim,), weights_mean, device=device),
            torch.tensor([bias_mean], device=device),
        )
    )
    sut.set_parameters(parameters)
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([5.0])),
        (torch.tensor([[0.0, 0.0]]), torch.tensor([1.0])),
        (torch.tensor([[-1.0, -1.0]]), torch.tensor([-3.0])),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([3.0])),
        (torch.tensor([[-1.0, 0.0]]), torch.tensor([-1.0])),
        (torch.tensor([[-1.0, 1.0]]), torch.tensor([1.0])),
        (
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]),
            torch.tensor([5.0, 1.0, -3.0]),
        ),
    ],
)
def test_zero_mean_two_inputs(inputs: Tensor, expected: Tensor) -> None:
    input_dim = 2
    sut = LinearMean(input_dim, device=device)
    parameters = torch.concat(
        (
            torch.full((input_dim,), weights_mean, device=device),
            torch.tensor([bias_mean], device=device),
        )
    )
    sut.set_parameters(parameters)
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)
