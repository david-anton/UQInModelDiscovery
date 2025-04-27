from typing import TypeAlias

import torch

from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.normalizingflows.flows.normalizingflows import (
    NormalizingFlowProtocol,
)
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)

Samples: TypeAlias = list[Tensor]


def sample_from_normalizing_flow(
    normalizing_flow: NormalizingFlowProtocol, num_samples: int
) -> tuple[MomentsMultivariateNormal, NPArray]:

    def draw_samples() -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples)
        return list(samples)

    samples_list = draw_samples()
    return determine_statistical_moments(samples_list)


def determine_statistical_moments(
    samples_list: Samples,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    moments = determine_moments_of_multivariate_normal_distribution(samples)
    return moments, samples
