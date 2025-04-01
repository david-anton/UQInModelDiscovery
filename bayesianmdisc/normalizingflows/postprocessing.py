from typing import TypeAlias

import torch

from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)

Samples: TypeAlias = list[Tensor]


def determine_statistical_moments(
    samples_list: Samples,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    moments = determine_moments_of_multivariate_normal_distribution(samples)
    return moments, samples
