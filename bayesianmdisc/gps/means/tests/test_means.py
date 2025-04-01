import pytest
import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.gps.means import ZeroMean

device = torch.device("cpu")
constant_mean = 2.0


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([1.0]), torch.tensor([0.0])),
        (torch.tensor([0.0]), torch.tensor([0.0])),
        (torch.tensor([-1.0]), torch.tensor([0.0])),
        (torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])),
    ],
)
def test_zero_mean(inputs: Tensor, expected: Tensor) -> None:
    sut = ZeroMean(device)
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)
