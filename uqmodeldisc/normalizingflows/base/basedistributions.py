from typing import Iterator, Protocol

from uqmodeldisc.customtypes import Parameter, Tensor


class BaseDistributionProtocol(Protocol):
    def forward(self, num_samples: int = 1) -> tuple[Tensor, Tensor]:
        pass

    def sample(self, num_samples: int = 1) -> Tensor:
        pass

    def log_prob(self, base_samples: Tensor) -> Tensor:
        pass

    def __call__(self, num_samples: int) -> Tensor:
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        pass
