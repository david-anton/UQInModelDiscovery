import torch

from bayesianmdisc.customtypes import Tensor, TensorSize


def logarithmic_sum_of_exponentials(log_probs: Tensor) -> Tensor:
    max_log_prob = torch.amax(log_probs, dim=0)
    return max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob), dim=0)
    )


def flatten_tensor(tensor: Tensor) -> Tensor:
    if tensor.dim() == 1:
        return tensor
    return torch.transpose(tensor, 1, 0).ravel()


def repeat_tensor(tensor: Tensor, repeat_size: TensorSize) -> Tensor:
    return tensor.repeat(repeat_size)


def concat_zero_dimensional_tensors(tensors: tuple[Tensor, ...]) -> Tensor:
    unsqueezed_tensors = [tensor.unsqueeze(dim=0) for tensor in tensors]
    return torch.concat(unsqueezed_tensors, dim=0)


def unsqueeze_if_necessary(tensor: Tensor) -> Tensor:
    num_dims = tensor.dim()
    if num_dims == 0:
        return torch.unsqueeze(tensor, dim=0)
    else:
        return tensor
