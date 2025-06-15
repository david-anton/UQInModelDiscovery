from collections import namedtuple
from typing import TypeAlias

import numpy as np
import torch

from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.errors import StatisticsError

QuantileValues: TypeAlias = tuple[NPArray, NPArray]

MomentsUnivariateNormal = namedtuple(
    "MomentsUnivariateNormal", ["mean", "standard_deviation"]
)


def determine_moments_of_univariate_normal_distribution(
    samples: NPArray,
) -> MomentsUnivariateNormal:
    mean = np.mean(samples, axis=0)
    standard_deviation = np.std(samples, axis=0, ddof=1)
    return MomentsUnivariateNormal(mean=mean, standard_deviation=standard_deviation)


MomentsMultivariateNormal = namedtuple(
    "MomentsMultivariateNormal", ["mean", "covariance"]
)


def determine_moments_of_multivariate_normal_distribution(
    samples: NPArray,
) -> MomentsMultivariateNormal:
    mean = np.mean(samples, axis=0, keepdims=False)
    covariance = np.cov(samples, rowvar=False)
    if covariance.shape == ():
        covariance = np.array([covariance])
    return MomentsMultivariateNormal(mean=mean, covariance=covariance)


def logarithmic_sum_of_exponentials(log_probs: Tensor) -> Tensor:
    max_log_prob = torch.amax(log_probs, dim=0)
    return max_log_prob + torch.log(
        torch.sum(torch.exp(log_probs - max_log_prob), dim=0)
    )


def determine_quantiles_from_samples(
    samples: NPArray, credible_interval: float
) -> QuantileValues:

    def _validate_credible_interval(credible_interval: float) -> None:
        is_larger_or_equal_zero = credible_interval >= 0.0
        is_smaller_or_equal_one = credible_interval <= 1.0
        is_valid = is_larger_or_equal_zero and is_smaller_or_equal_one

        if not is_valid:
            raise StatisticsError(
                f"""The credible interval is expected to be positive 
                    and smaller or equal than one, but is {credible_interval}"""
            )

    _validate_credible_interval(credible_interval)

    quantile = (1.0 - credible_interval) / 2
    min_quantile = quantile
    max_quantile = 1.0 - quantile
    return np.quantile(
        samples, [min_quantile, max_quantile], method="inverted_cdf", axis=0
    )
