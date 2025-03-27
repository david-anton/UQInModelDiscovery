import torch
from torch.func import vmap

from bayesianmdisc.bayes.prior import Prior
from bayesianmdisc.types import Device, Tensor


class BaseDistributionWrapper(torch.nn.Module):
    def __init__(self, prior: Prior, device: Device) -> None:
        super().__init__()
        self._prior = prior
        self._device = device

    def forward(self, num_samples: int = 1) -> tuple[Tensor, Tensor]:
        samples = self.sample(num_samples)
        log_probs = self.log_prob(samples)
        return samples, log_probs

    def sample(self, num_samples: int = 1) -> Tensor:
        return self._prior.sample(num_samples=num_samples)

    def log_prob(self, base_samples: Tensor) -> Tensor:
        def vmap_log_prob_func(sample: Tensor) -> Tensor:
            return self._prior.log_prob_with_grad(sample)

        if base_samples.dim() == 1:
            return vmap_log_prob_func(base_samples)
        else:
            return vmap(vmap_log_prob_func)(base_samples)
