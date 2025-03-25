import torch

from bayesianmdisc.types import Tensor


def logarithmic_sum_of_exponentials(log_probs: Tensor) -> Tensor:
    max_log_prob = torch.amax(log_probs, dim=0)
    return max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob), dim=0)
    )
