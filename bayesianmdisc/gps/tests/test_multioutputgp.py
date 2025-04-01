import pytest
import torch

from bayesianmdisc.gps.multioutputgp import flatten_outputs
from bayesianmdisc.customtypes import Tensor


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (
            torch.tensor([1.0, 10.0, 100.0]),
            torch.tensor([1.0, 10.0, 100.0]),
        ),
        (
            torch.tensor([[1.0], [10.0], [100.0]]),
            torch.tensor([1.0, 10.0, 100.0]),
        ),
        (
            torch.tensor([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]),
            torch.tensor([1.0, 10.0, 100.0, 2.0, 20.0, 200.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]),
            torch.tensor([1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0]),
        ),
    ],
)
def test_flatten_outputs(outputs: Tensor, expected: Tensor) -> None:
    sut = flatten_outputs

    actual = sut(outputs)

    torch.testing.assert_close(actual, expected)
