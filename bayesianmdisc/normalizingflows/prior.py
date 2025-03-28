from bayesianmdisc.normalizingflows.flows import NormalizingFlowProtocol

import torch

from bayesianmdisc.types import Tensor, Device


class NormalizingFlowPrior:
    def __init__(
        self, normalizing_flow: NormalizingFlowProtocol, dim: int, device: Device
    ) -> None:
        self.dim = dim
        self._normalizing_flow = normalizing_flow
        self._device = device

    def prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._log_prob(parameters)

    def log_prob_with_grad(self, parameters: Tensor) -> Tensor:
        return self._log_prob(parameters)

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        return torch.autograd.grad(
            self._log_prob(parameters),
            parameters,
            retain_graph=False,
            create_graph=False,
        )[0]

    def sample(self, num_samples: int = 1) -> Tensor:
        samples, _ = self._normalizing_flow.sample(num_samples)
        return samples

    def _prob(self, parameters: Tensor) -> Tensor:
        return torch.exp(self._log_prob(parameters))

    def _log_prob(self, parameters: Tensor) -> Tensor:
        return self._normalizing_flow.log_prob(parameters)
