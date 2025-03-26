from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from bayesianmdisc.networks.ffnn import FFNN

layer_sizes = [2, 4, 1]
num_nn_parameters = 8 + 4 + 4 + 1


@pytest.fixture
def sut() -> FFNN:
    return FFNN(
        layer_sizes=layer_sizes,
        activation=torch.nn.Identity(),
        init_weights=nn.init.ones_,
        init_bias=nn.init.ones_,
    )


def test_forward_one_input(sut: FFNN) -> None:
    input = torch.tensor([1.0, 2.0])

    actual = sut(input)

    expected = torch.tensor([17.0])
    torch.testing.assert_close(actual, expected)


def test_forward_multiple_input(sut: FFNN) -> None:
    input = torch.tensor([[1.0, 2.0], [1.0, 2.0]])

    actual = sut(input)

    expected = torch.tensor([[17.0], [17.0]])
    torch.testing.assert_close(actual, expected)
