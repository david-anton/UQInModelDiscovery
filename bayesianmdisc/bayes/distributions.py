from typing import Protocol, TypeAlias, Union

import torch

from bayesianmdisc.customtypes import Device, Tensor, NPArray
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateGammaDistribution as _IndependentMultivariateGammaDistribution,
)
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateHalfNormalDistribution as _IndependentMultivariateHalfNormalDistribution,
)
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateInverseGammaDistribution as _IndependentMultivariateInverseGammaDistribution,
)
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateNormalDistribution as _IndependentMultivariateNormalDistribution,
)
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateStudentTDistribution as _IndependentMultivariateStudentTDistribution,
)
from bayesianmdisc.statistics.distributions import (
    MultivariateNormalDistribution as _MultivariateNormalDistribution,
)
from bayesianmdisc.statistics.distributions import (
    MultivariateUniformDistribution as _MultivariateUniformDistribution,
)
from bayesianmdisc.statistics.distributions import (
    UnivariateGammaDistribution as _UnivariateGammaDistribution,
)
from bayesianmdisc.statistics.distributions import (
    UnivariateHalfNormalDistribution as _UnivariateHalfNormalDistribution,
)
from bayesianmdisc.statistics.distributions import (
    UnivariateInverseGammaDistribution as _UnivariateInverseGammaDistribution,
)
from bayesianmdisc.statistics.distributions import (
    UnivariateNormalDistribution as _UnivariateNormalDistribution,
)
from bayesianmdisc.statistics.distributions import (
    UnivariateUniformDistribution as _UnivariateUniformDistribution,
)
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_gamma_distribution as _create_independent_multivariate_gamma_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_half_normal_distribution as _create_independent_multivariate_half_normal_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_inverse_gamma_distribution as _create_independent_multivariate_inverse_gamma_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_normal_distribution as _create_independent_multivariate_normal_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_studentT_distribution as _create_independent_multivariate_studentT_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_multivariate_normal_distribution as _create_multivariate_normal_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_multivariate_uniform_distribution as _create_multivariate_uniform_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_univariate_gamma_distribution as _create_univariate_gamma_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_univariate_half_normal_distribution as _create_univariate_half_normal_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_univariate_inverse_gamma_distribution as _create_univariate_inverse_gamma_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_univariate_normal_distribution as _create_univariate_normal_distribution,
)
from bayesianmdisc.statistics.distributions import (
    create_univariate_uniform_distribution as _create_univariate_uniform_distribution,
)
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)

_Distribution: TypeAlias = Union[
    _UnivariateUniformDistribution,
    _UnivariateNormalDistribution,
    _UnivariateHalfNormalDistribution,
    _MultivariateNormalDistribution,
    _MultivariateUniformDistribution,
    _IndependentMultivariateNormalDistribution,
    _IndependentMultivariateGammaDistribution,
    _IndependentMultivariateInverseGammaDistribution,
    _UnivariateGammaDistribution,
    _UnivariateInverseGammaDistribution,
    _IndependentMultivariateStudentTDistribution,
    _IndependentMultivariateHalfNormalDistribution,
]


class DistributionProtocol(Protocol):
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


class Distribution:
    def __init__(self, distribution: _Distribution):
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


class MultipliedDistributions(Distribution):
    def __init__(self, distributions: list[DistributionProtocol]):
        self._distributions = distributions
        self._distribution_dims = [distribution.dim for distribution in distributions]
        self.dim = sum(self._distribution_dims)

    def sample(self, num_samples: int = 1) -> Tensor:
        if num_samples == 1:
            samples = [
                distribution.sample(num_samples) for distribution in self._distributions
            ]
            return torch.concat(samples, dim=0)
        else:
            samples = [
                distribution.sample(num_samples) for distribution in self._distributions
            ]
            return torch.concat(samples, dim=1)

    def _log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = []
        start_index = 0
        for i, distribution in enumerate(self._distributions[:-1]):
            dim_parameters_i = self._distribution_dims[i]
            parameters_i = parameters[start_index : start_index + dim_parameters_i]
            log_probs.append(
                torch.unsqueeze(distribution.log_prob_with_grad(parameters_i), dim=0)
            )
            start_index += dim_parameters_i
        parameters_last = parameters[start_index:]
        log_probs.append(
            torch.unsqueeze(
                self._distributions[-1].log_prob_with_grad(parameters_last), dim=0
            )
        )
        return torch.sum(torch.concat(log_probs), dim=0)


def create_univariate_uniform_distribution(
    lower_limit: float, upper_limit: float, device: Device
) -> Distribution:
    distribution = _create_univariate_uniform_distribution(
        lower_limit, upper_limit, device
    )
    return Distribution(distribution)


def create_multivariate_uniform_distribution(
    lower_limits: Tensor, upper_limits: Tensor, device: Device
) -> Distribution:
    distribution = _create_multivariate_uniform_distribution(
        lower_limits, upper_limits, device
    )
    return Distribution(distribution)


def create_univariate_normal_distribution(
    mean: float, standard_deviation: float, device: Device
) -> Distribution:
    distribution = _create_univariate_normal_distribution(
        mean, standard_deviation, device
    )
    return Distribution(distribution)


def create_univariate_half_normal_distribution(
    standard_deviation: float, device: Device
) -> Distribution:
    distribution = _create_univariate_half_normal_distribution(
        standard_deviation, device
    )
    return Distribution(distribution)


def create_multivariate_normal_distribution(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> Distribution:
    distribution = _create_multivariate_normal_distribution(
        means, covariance_matrix, device
    )
    return Distribution(distribution)


def create_independent_multivariate_normal_distribution(
    means: Tensor, standard_deviations: Tensor, device: Device
) -> Distribution:
    distribution = _create_independent_multivariate_normal_distribution(
        means, standard_deviations, device
    )
    return Distribution(distribution)


def create_independent_multivariate_half_normal_distribution(
    standard_deviations: Tensor, device: Device
) -> Distribution:
    distribution = _create_independent_multivariate_half_normal_distribution(
        standard_deviations, device
    )
    return Distribution(distribution)


def create_univariate_gamma_distribution(
    concentration: float, rate: float, device: Device
) -> Distribution:
    distribution = _create_univariate_gamma_distribution(concentration, rate, device)
    return Distribution(distribution)


def create_independent_multivariate_gamma_distribution(
    concentrations: Tensor, rates: Tensor, device: Device
) -> Distribution:
    distribution = _create_independent_multivariate_gamma_distribution(
        concentrations, rates, device
    )
    return Distribution(distribution)


def create_univariate_inverse_gamma_distribution(
    concentration: float, rate: float, device: Device
) -> Distribution:
    distribution = _create_univariate_inverse_gamma_distribution(
        concentration, rate, device
    )
    return Distribution(distribution)


def create_independent_multivariate_inverse_gamma_distribution(
    concentrations: Tensor, rates: Tensor, device: Device
) -> Distribution:
    distribution = _create_independent_multivariate_inverse_gamma_distribution(
        concentrations, rates, device
    )
    return Distribution(distribution)


def create_independent_multivariate_studentT_distribution(
    degrees_of_freedom: Tensor, means: Tensor, scales: Tensor, device: Device
) -> Distribution:
    distribution = _create_independent_multivariate_studentT_distribution(
        degrees_of_freedom, means, scales, device
    )
    return Distribution(distribution)


def multiply_distributions(
    distributions: list[DistributionProtocol],
) -> MultipliedDistributions:
    return MultipliedDistributions(distributions)


def sample_and_analyse_distribution(
    distribution: DistributionProtocol, num_samples: int
) -> tuple[MomentsMultivariateNormal, NPArray]:

    def draw_samples() -> list[Tensor]:
        samples = distribution.sample(num_samples)
        return list(samples)

    samples_list = draw_samples()
    return _determine_statistical_moments(samples_list)


Samples: TypeAlias = list[Tensor]


def _determine_statistical_moments(
    samples_list: Samples,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    moments = determine_moments_of_multivariate_normal_distribution(samples)
    return moments, samples
