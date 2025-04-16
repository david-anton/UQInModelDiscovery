from collections import namedtuple

import numpy as np
import torch

from bayesianmdisc.customtypes import NPArray, Tensor

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
