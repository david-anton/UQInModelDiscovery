import torch

from bayesianmdisc.customtypes import Tensor


def flatten_outputs(outputs: Tensor) -> Tensor:
    if outputs.dim() == 1:
        return outputs
    else:
        return torch.transpose(outputs, 1, 0).ravel()
