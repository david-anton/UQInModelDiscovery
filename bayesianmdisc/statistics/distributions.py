from typing import Protocol

import torch

from bayesianmdisc.customtypes import Device, Tensor, TensorSize
from bayesianmdisc.errors import ProbabilityDistributionError


def squeeze_if_necessary(log_prob: Tensor) -> Tensor:
    if log_prob.shape is not torch.Size([]):
        return torch.squeeze(log_prob, dim=0)
    return log_prob


def define_samples_size(num_samples: int) -> TensorSize:
    if num_samples == 1:
        return torch.Size()
    else:
        return torch.Size((num_samples,))


def reshape_univariate_distribution_samples(samples: Tensor) -> Tensor:
    if torch.numel(samples) == 1:
        return torch.unsqueeze(samples, dim=0)
    else:
        return samples.reshape((-1, 1))


class Distribution(Protocol):
    def log_prob(self, sample: Tensor) -> Tensor:
        pass

    def sample(self, num_samples: int = 1) -> Tensor:
        pass


class UnivariateUniformDistribution:
    def __init__(self, lower_limit: float, upper_limit: float, device: Device):
        self._distribution = torch.distributions.Uniform(
            low=torch.tensor(lower_limit, device=device),
            high=torch.tensor(upper_limit, device=device),
            validate_args=False,
        )
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return reshape_univariate_distribution_samples(
            self._distribution.rsample(sample_shape)
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class MultivariateUniformDistribution:
    def __init__(self, lower_limits: Tensor, upper_limits: Tensor, device: Device):
        self._validate_limits(lower_limits, upper_limits)
        self._distribution = torch.distributions.Uniform(
            low=lower_limits.to(device),
            high=upper_limits.to(device),
            validate_args=False,
        )
        self.dim = torch.numel(lower_limits)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = torch.sum(self._distribution.log_prob(sample), dim=0)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_limits(self, lower_limits: Tensor, upper_limits: Tensor) -> None:
        shape_lower_limits = lower_limits.size()
        shape_upper_limits = upper_limits.size()
        if not shape_lower_limits == shape_upper_limits:
            raise ProbabilityDistributionError(
                f"""Different shape of lower and upper limits: 
                {shape_lower_limits} and {shape_upper_limits}"""
            )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class UnivariateNormalDistribution:
    def __init__(self, mean: float, standard_deviation: float, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=torch.tensor([mean], device=device),
            covariance_matrix=torch.tensor([[standard_deviation**2]], device=device),
            validate_args=False,
        )
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return reshape_univariate_distribution_samples(
            self._distribution.rsample(sample_shape)
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class UnivariateHalfNormalDistribution:
    def __init__(self, standard_deviation: float, device: Device):
        self._distribution = torch.distributions.HalfNormal(
            scale=torch.tensor([standard_deviation], device=device),
            validate_args=False,
        )
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return reshape_univariate_distribution_samples(
            self._distribution.rsample(sample_shape)
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class MultivariateNormalDistribution:
    def __init__(self, means: Tensor, covariance_matrix: Tensor, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=means.to(device),
            covariance_matrix=covariance_matrix.to(device),
            validate_args=False,
        )
        self.means = self._distribution.mean
        self.variances = self._distribution.variance
        self.dim = torch.numel(means)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class IndependentMultivariateNormalDistribution:
    def __init__(self, means: Tensor, standard_deviations: Tensor, device: Device):
        self._distribution = torch.distributions.Normal(
            loc=means.to(device),
            scale=standard_deviations.to(device),
            validate_args=False,
        )
        self.means = self._distribution.mean
        self.standard_deviations = self._distribution.stddev
        self.dim = torch.numel(means)

    def log_probs_individual(self, sample: Tensor) -> Tensor:
        return self._distribution.log_prob(sample)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_probs_individual = self.log_probs_individual(sample)
        log_prob = torch.sum(log_probs_individual)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class IndependentMultivariateHalfNormalDistribution:
    def __init__(self, standard_deviations: Tensor, device: Device):
        self._distribution = torch.distributions.HalfNormal(
            scale=standard_deviations.to(device),
            validate_args=False,
        )
        self.standard_deviations = self._distribution.stddev
        self.dim = torch.numel(standard_deviations)

    def log_probs_individual(self, sample: Tensor) -> Tensor:
        return self._distribution.log_prob(sample)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_probs_individual = self.log_probs_individual(sample)
        log_prob = torch.sum(log_probs_individual)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class UnivariateGammaDistribution:
    def __init__(self, concentration: float, rate: float, device: Device) -> None:
        self._distribution = torch.distributions.Gamma(
            concentration=torch.tensor(concentration, device=device),
            rate=torch.tensor(rate, device=device),
            validate_args=False,
        )
        self._device = device
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return reshape_univariate_distribution_samples(
            self._distribution.rsample(sample_shape)
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class IndependentMultivariateGammaDistribution:
    def __init__(self, concentrations: Tensor, rates: Tensor, device: Device) -> None:
        self._distribution = torch.distributions.Gamma(
            concentration=concentrations.to(device),
            rate=rates.to(device),
            validate_args=False,
        )
        self._device = device
        self.dim = torch.numel(concentrations)

    def log_probs_individual(self, sample: Tensor) -> Tensor:
        return self._distribution.log_prob(sample)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_probs_individual = self.log_probs_individual(sample)
        log_prob = torch.sum(log_probs_individual)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class UnivariateInverseGammaDistribution:
    def __init__(self, concentration: float, rate: float, device: Device) -> None:
        self._distribution = torch.distributions.InverseGamma(
            concentration=torch.tensor(concentration, device=device),
            rate=torch.tensor(rate, device=device),
            validate_args=False,
        )
        self._device = device
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return reshape_univariate_distribution_samples(
            self._distribution.rsample(sample_shape)
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


class IndependentMultivariateStudentTDistribution:
    def __init__(
        self, degrees_of_freedom: Tensor, means: Tensor, scales: Tensor, device: Device
    ):
        self._distribution = torch.distributions.StudentT(
            df=degrees_of_freedom.to(device),
            loc=means.to(device),
            scale=scales.to(device),
            validate_args=False,
        )
        self.degrees_of_freedom = self._distribution.df
        self.means = self._distribution.mean
        self.scales = self._distribution.scale
        self.dim = torch.numel(means)

    def log_probs_individual(self, sample: Tensor) -> Tensor:
        return self._distribution.log_prob(sample)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_probs_individual = self.log_probs_individual(sample)
        log_prob = torch.sum(log_probs_individual)
        return squeeze_if_necessary(log_prob)

    def sample(self, num_samples: int = 1) -> Tensor:
        sample_shape = define_samples_size(num_samples)
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise ProbabilityDistributionError(f"Unexpected shape of sample: {shape}.")


def create_univariate_uniform_distribution(
    lower_limit: float, upper_limit: float, device: Device
) -> UnivariateUniformDistribution:
    return UnivariateUniformDistribution(lower_limit, upper_limit, device)


def create_multivariate_uniform_distribution(
    lower_limits: Tensor, upper_limits: Tensor, device: Device
) -> MultivariateUniformDistribution:
    return MultivariateUniformDistribution(lower_limits, upper_limits, device)


def create_univariate_normal_distribution(
    mean: float, standard_deviation: float, device: Device
) -> UnivariateNormalDistribution:
    return UnivariateNormalDistribution(mean, standard_deviation, device)


def create_univariate_half_normal_distribution(
    standard_deviation: float, device: Device
) -> UnivariateHalfNormalDistribution:
    return UnivariateHalfNormalDistribution(standard_deviation, device)


def create_multivariate_normal_distribution(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> MultivariateNormalDistribution:
    return MultivariateNormalDistribution(means, covariance_matrix, device)


def create_independent_multivariate_normal_distribution(
    means: Tensor, standard_deviations: Tensor, device: Device
) -> IndependentMultivariateNormalDistribution:
    return IndependentMultivariateNormalDistribution(means, standard_deviations, device)


def create_independent_multivariate_half_normal_distribution(
    standard_deviations: Tensor, device: Device
) -> IndependentMultivariateHalfNormalDistribution:
    return IndependentMultivariateHalfNormalDistribution(standard_deviations, device)


def create_univariate_gamma_distribution(
    concentration: float, rate: float, device: Device
) -> UnivariateGammaDistribution:
    return UnivariateGammaDistribution(concentration, rate, device)


def create_independent_multivariate_gamma_distribution(
    concentrations: Tensor, rates: Tensor, device: Device
) -> IndependentMultivariateGammaDistribution:
    return IndependentMultivariateGammaDistribution(
        concentrations=concentrations, rates=rates, device=device
    )


def create_univariate_inverse_gamma_distribution(
    concentration: float, rate: float, device: Device
) -> UnivariateInverseGammaDistribution:
    return UnivariateInverseGammaDistribution(concentration, rate, device)


def create_independent_multivariate_studentT_distribution(
    degrees_of_freedom: Tensor, means: Tensor, scales: Tensor, device: Device
) -> IndependentMultivariateStudentTDistribution:
    return IndependentMultivariateStudentTDistribution(
        degrees_of_freedom, means, scales, device
    )
