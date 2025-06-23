import torch

from bayesianmdisc.customtypes import Tensor, NPArray, Device


def flatten_outputs(outputs: Tensor) -> Tensor:
    if outputs.dim() == 1:
        return outputs
    else:
        return torch.transpose(outputs, 1, 0).ravel()


def from_numpy_to_torch(array: NPArray, device: Device) -> Tensor:
    return torch.from_numpy(array).type(torch.get_default_dtype()).to(device)


def from_torch_to_numpy(tensor: Tensor) -> NPArray:
    return tensor.detach().cpu().numpy()
