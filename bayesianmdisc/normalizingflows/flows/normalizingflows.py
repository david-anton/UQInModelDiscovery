from typing import Any, Dict, Iterator, Mapping, Protocol, TypeVar

import torch

from bayesianmdisc.customtypes import (
    Device,
    NFBaseDistribution,
    NFNormalizingFlow,
    Parameter,
    Tensor,
)

T = TypeVar("T", bound="NormalizingFlowProtocol")


class NormalizingFlowProtocol(Protocol):
    dim: int

    def forward(self, base_samples: Tensor) -> Tensor:
        pass

    def forward_u_and_sum_log_det_u(
        self, base_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass

    def inverse(self, samples: Tensor) -> Tensor:
        pass

    def inverse_x_and_sum_log_det_x(self, samples: Tensor) -> tuple[Tensor, Tensor]:
        pass

    def sample(self, num_samples: int = 1) -> tuple[Tensor, Tensor]:
        pass

    def log_prob(self, samples: Tensor) -> Tensor:
        pass

    def print_summary(self) -> None:
        pass

    def __call__(self, num_samples: int) -> Tensor:
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        pass

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        pass

    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        pass

    def train(self: T) -> T:
        pass

    def eval(self: T) -> T:
        pass


class NormalizingFlow(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        flows: list[NFNormalizingFlow],
        base_distribution: NFBaseDistribution,
        device: Device,
    ) -> None:
        super().__init__()
        self.dim = dimension
        self.flows = torch.nn.ModuleList(flows)
        self.base_distribution = base_distribution
        self._device = device

    def forward(self, base_samples: Tensor) -> Tensor:
        x, _ = self.forward_u_and_sum_log_det_u(base_samples)
        return x

    def forward_u_and_sum_log_det_u(
        self, base_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        u = base_samples
        sum_log_det_u = torch.zeros(len(u), device=self._device)

        for sub_flow in self.flows:
            u, log_det_u = sub_flow(u)
            sum_log_det_u = sum_log_det_u + log_det_u

        # The returned u is equal to x.
        return u, sum_log_det_u

    def inverse(self, samples: Tensor) -> Tensor:
        samples = self._unsqueeze_single_sample(samples)
        u, _ = self.inverse_x_and_sum_log_det_x(samples)
        return u

    def inverse_x_and_sum_log_det_x(self, samples: Tensor) -> tuple[Tensor, Tensor]:
        x = self._unsqueeze_single_sample(samples)
        sum_log_det_x = torch.zeros(len(x), device=self._device)

        for sub_flow in reversed(self.flows):
            x, log_det_x = sub_flow.inverse(x)
            sum_log_det_x = sum_log_det_x + log_det_x

        return x, sum_log_det_x

    def sample(self, num_samples: int = 1) -> tuple[Tensor, Tensor]:
        u, log_prob_u = self.base_distribution(num_samples)

        x, sum_log_det_x = self.forward_u_and_sum_log_det_u(u)
        log_prob_x = log_prob_u - sum_log_det_x

        return x, log_prob_x

    def log_prob(self, samples: Tensor) -> Tensor:
        x = self._unsqueeze_single_sample(samples)
        u, sum_log_det_u = self.inverse_x_and_sum_log_det_x(x)
        log_prob = self.base_distribution.log_prob(u) + sum_log_det_u
        return self._squeeze_log_prob(log_prob)

    def print_summary(self) -> None:
        num_layers = len(self.flows)
        num_total_parameters = 0
        num_trainable_parameters = 0

        for parameters in self.parameters():
            num_parameters = parameters.numel()
            num_total_parameters += num_parameters
            if parameters.requires_grad:
                num_trainable_parameters += num_parameters

        print("############################################################")
        print("Normalizing flow summary:")
        print(f"Number layers: {num_layers}")
        print(f"Number total parameters: {num_total_parameters}")
        print(f"Number trainable parameters: {num_trainable_parameters}")
        print("############################################################")

    def _unsqueeze_single_sample(self, samples: Tensor) -> Tensor:
        if samples.dim() == 1:
            return torch.unsqueeze(samples, dim=0)
        return samples

    def _squeeze_log_prob(self, log_prob: Tensor) -> Tensor:
        return torch.squeeze(log_prob, dim=0)
