from typing import Protocol, TypeAlias, Union

import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateGammaDistribution,
    IndependentMultivariateNormalDistribution,
    IndependentMultivariateStudentTDistribution,
    MultivariateNormalDistribution,
    MultivariateUniformDistribution,
    UnivariateGammaDistribution,
    UnivariateHalfNormalDistribution,
    UnivariateInverseGammaDistribution,
    UnivariateNormalDistribution,
    UnivariateUniformDistribution,
    create_independent_multivariate_gamma_distribution,
    create_independent_multivariate_normal_distribution,
    create_independent_multivariate_studentT_distribution,
    create_multivariate_normal_distribution,
    create_multivariate_uniform_distribution,
    create_univariate_gamma_distribution,
    create_univariate_half_normal_distribution,
    create_univariate_inverse_gamma_distribution,
    create_univariate_normal_distribution,
    create_univariate_uniform_distribution,
)

PriorDistribution: TypeAlias = Union[
    UnivariateUniformDistribution,
    UnivariateNormalDistribution,
    UnivariateHalfNormalDistribution,
    MultivariateNormalDistribution,
    MultivariateUniformDistribution,
    IndependentMultivariateNormalDistribution,
    IndependentMultivariateGammaDistribution,
    UnivariateGammaDistribution,
    UnivariateInverseGammaDistribution,
    IndependentMultivariateStudentTDistribution,
]


class PriorProtocol(Protocol):
    dim: int

    def prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob_with_grad(self, parameters: Tensor) -> Tensor:
        pass

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def sample(self, num_samples: int = 1) -> Tensor:
        pass


class Prior:
    def __init__(self, distribution: PriorDistribution):
        self.distribution = distribution
        self.dim = distribution.dim

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
        return self.distribution.sample(num_samples)

    def _prob(self, parameters: Tensor) -> Tensor:
        return torch.exp(self._log_prob(parameters))

    def _log_prob(self, parameters: Tensor) -> Tensor:
        return self.distribution.log_prob(parameters)


class MultipliedPriors(Prior):
    def __init__(self, priors: list[Prior]):
        self._priors = priors
        self._prior_dims = [prior.dim for prior in priors]
        self.dim = sum(self._prior_dims)

    def sample(self, num_samples: int = 1) -> Tensor:
        if num_samples == 1:
            samples = [prior.sample(num_samples) for prior in self._priors]
            return torch.concat(samples, dim=0)
        else:
            samples = [prior.sample(num_samples) for prior in self._priors]
            return torch.concat(samples, dim=1)

    def _log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = []
        start_index = 0
        for i, prior in enumerate(self._priors[:-1]):
            dim_parameters_i = self._prior_dims[i]
            parameters_i = parameters[start_index : start_index + dim_parameters_i]
            log_probs.append(torch.unsqueeze(prior._log_prob(parameters_i), dim=0))
            start_index += dim_parameters_i
        parameters_last = parameters[start_index:]
        log_probs.append(
            torch.unsqueeze(self._priors[-1]._log_prob(parameters_last), dim=0)
        )
        return torch.sum(torch.concat(log_probs), dim=0)


def create_univariate_uniform_distributed_prior(
    lower_limit: float, upper_limit: float, device: Device
) -> Prior:
    distribution = create_univariate_uniform_distribution(
        lower_limit, upper_limit, device
    )
    return Prior(distribution)


def create_multivariate_uniform_distributed_prior(
    lower_limits: Tensor, upper_limits: Tensor, device: Device
) -> Prior:
    distribution = create_multivariate_uniform_distribution(
        lower_limits, upper_limits, device
    )
    return Prior(distribution)


def create_univariate_normal_distributed_prior(
    mean: float, standard_deviation: float, device: Device
) -> Prior:
    distribution = create_univariate_normal_distribution(
        mean, standard_deviation, device
    )
    return Prior(distribution)


def create_univariate_half_normal_distributed_prior(
    standard_deviation: float, device: Device
) -> Prior:
    distribution = create_univariate_half_normal_distribution(
        standard_deviation, device
    )
    return Prior(distribution)


def create_multivariate_normal_distributed_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> Prior:
    distribution = create_multivariate_normal_distribution(
        means, covariance_matrix, device
    )
    return Prior(distribution)


def create_independent_multivariate_normal_distributed_prior(
    means: Tensor, standard_deviations: Tensor, device: Device
) -> Prior:
    distribution = create_independent_multivariate_normal_distribution(
        means, standard_deviations, device
    )
    return Prior(distribution)


def create_univariate_gamma_distributed_prior(
    concentration: float, rate: float, device: Device
) -> Prior:
    distribution = create_univariate_gamma_distribution(concentration, rate, device)
    return Prior(distribution)


def create_independent_multivariate_gamma_distributed_prior(
    concentrations: Tensor, rates: Tensor, device: Device
) -> Prior:
    distribution = create_independent_multivariate_gamma_distribution(
        concentrations, rates, device
    )
    return Prior(distribution)


def create_univariate_inverse_gamma_distributed_prior(
    concentration: float, rate: float, device: Device
) -> Prior:
    distribution = create_univariate_inverse_gamma_distribution(
        concentration, rate, device
    )
    return Prior(distribution)


def create_independent_multivariate_studentT_distributed_prior(
    degrees_of_freedom: Tensor, means: Tensor, scales: Tensor, device: Device
) -> Prior:
    distribution = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom, means, scales, device
    )
    return Prior(distribution)


def multiply_priors(priors: list[Prior]) -> MultipliedPriors:
    return MultipliedPriors(priors)
