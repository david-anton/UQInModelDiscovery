import pytest
import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.gps.means.linearmean import LinearMean

device = torch.device("cpu")
weights_mean = 2.0


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([0.0]), torch.tensor([0.0])),
        (torch.tensor([-1.0]), torch.tensor([-2.0])),
        (torch.tensor([1.0, 0.0, -1.0]), torch.tensor([2.0, 0.0, -2.0])),
        (torch.tensor([[1.0], [0.0], [-1.0]]), torch.tensor([2.0, 0.0, -2.0])),
    ],
)
def test_linear_mean_one_dimensional_input(inputs: Tensor, expected: Tensor) -> None:
    input_dim = 1
    sut = LinearMean(input_dim, device=device)
    sut.set_parameters(torch.full((input_dim,), weights_mean, device=device))
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([4.0])),
        (torch.tensor([[0.0, 0.0]]), torch.tensor([0.0])),
        (torch.tensor([[-1.0, -1.0]]), torch.tensor([-4.0])),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([2.0])),
        (torch.tensor([[-1.0, 0.0]]), torch.tensor([-2.0])),
        (torch.tensor([[-1.0, 1.0]]), torch.tensor([0.0])),
        (
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]),
            torch.tensor([4.0, 0.0, -4.0]),
        ),
    ],
)
def test_zero_mean_two_inputs(inputs: Tensor, expected: Tensor) -> None:
    input_dim = 2
    sut = LinearMean(input_dim, device=device)
    sut.set_parameters(torch.full((input_dim,), weights_mean, device=device))
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)
