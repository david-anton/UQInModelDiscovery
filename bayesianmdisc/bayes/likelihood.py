from abc import ABC, abstractmethod
from typing import Optional, Protocol, TypeAlias

import torch
from torch import vmap

from bayesianmdisc.bayes.utility import flatten_tensor, unsqueeze_if_necessary
from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.errors import LikelihoodError
from bayesianmdisc.models import ModelProtocol
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateNormalDistribution,
    create_independent_multivariate_normal_distribution,
)

Prob: TypeAlias = Tensor
LogProb: TypeAlias = Tensor
GradLogProb: TypeAlias = Tensor
ErrorDIstribution: TypeAlias = IndependentMultivariateNormalDistribution


class LikelihoodProtocol(Protocol):
    model: ModelProtocol

    def prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob_with_grad(self, parameters: Tensor) -> Tensor:
        pass

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        pass


def calculate_error(y_predicted: Tensor, y_true: Tensor) -> Tensor:
    return y_predicted - y_true


class ErrorCalculator:
    def calculate(self, y_predicted: Tensor, y_true: Tensor) -> Tensor:
        return calculate_error(y_predicted, y_true)


class ErrorDistributionCreator:
    def __init__(self, device: Device) -> None:
        self._device = device

    def create(
        self, noise_stddevs: Tensor
    ) -> IndependentMultivariateNormalDistribution:
        dim = self._determine_dimension(noise_stddevs)
        means = self._assemble_means(dim)
        stddevs = noise_stddevs.to(self._device)
        return create_independent_multivariate_normal_distribution(
            means, stddevs, self._device
        )

    def _assemble_means(self, dim: int) -> Tensor:
        return torch.zeros((dim,), device=self._device)

    def _determine_dimension(self, noise_stddevs: Tensor) -> int:
        return len(noise_stddevs)


class LikelihoodBase(ABC):
    def __init__(
        self,
        model: ModelProtocol,
        inputs: Tensor,
        test_cases: Tensor,
        outputs: Tensor,
        min_noise_stddev: float,
        device: Device,
    ) -> None:
        self._validate_data(inputs, outputs)
        self.model = model
        self._inputs = inputs
        self._test_cases = test_cases
        self._true_outputs = outputs
        self._num_outputs = len(outputs)
        self._flattened_true_outputs = flatten_tensor(outputs)
        self._device = device
        self._error_calculator = ErrorCalculator()
        self._min_noise_stddev = torch.tensor(min_noise_stddev, device=self._device)
        self._error_distribution_creator = ErrorDistributionCreator(self._device)

    def prob(self, parameters: Tensor) -> Prob:
        with torch.no_grad():
            return self._prob(parameters)

    def log_prob(self, parameters: Tensor) -> LogProb:
        with torch.no_grad():
            return self._log_prob(parameters)

    def log_prob_with_grad(self, parameters: Tensor) -> LogProb:
        return self._log_prob(parameters)

    def grad_log_prob(self, parameters: Tensor) -> GradLogProb:
        return torch.autograd.grad(
            self._log_prob(parameters),
            parameters,
            retain_graph=True,
            create_graph=False,
        )[0]

    def _validate_data(self, inputs: Tensor, outputs: Tensor) -> None:
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        if not num_inputs == num_outputs:
            raise LikelihoodError(
                f"""The number of inputs and outputs is expected to be the same, 
                                  but is {num_inputs} and {num_outputs}."""
            )

    def _create_error_distribution(
        self, relative_noise_stddev: Tensor
    ) -> ErrorDIstribution:
        noise_stddev = self._assemble_noise_stddevs(relative_noise_stddev)
        return self._error_distribution_creator.create(noise_stddev)

    def _assemble_noise_stddevs(self, relative_noise_stddev: Tensor) -> Tensor:
        noise_stddevs = relative_noise_stddev * self._flattened_true_outputs
        return torch.where(
            noise_stddevs < self._min_noise_stddev,
            self._min_noise_stddev,
            noise_stddevs,
        )

    def _prob(self, parameters: Tensor) -> Prob:
        return torch.exp(self._log_prob(parameters))

    @abstractmethod
    def _log_prob(self, parameters: Tensor) -> LogProb:
        raise NotImplementedError()

    def _calculate_flattened_errors(self, parameters: Tensor) -> Tensor:
        outputs = self.model(self._inputs, self._test_cases, parameters)
        flattened_outputs = flatten_tensor(outputs)
        return self._error_calculator.calculate(
            flattened_outputs, self._flattened_true_outputs
        )


class LikelihoodFixedNoise(LikelihoodBase):
    def __init__(
        self,
        model: ModelProtocol,
        relative_noise_stddev: float,
        min_noise_stddev: float,
        inputs: Tensor,
        test_cases: Tensor,
        outputs: Tensor,
        device: Device,
    ) -> None:
        super().__init__(
            model=model,
            inputs=inputs,
            test_cases=test_cases,
            outputs=outputs,
            min_noise_stddev=min_noise_stddev,
            device=device,
        )
        self._error_distribution = self._create_error_distribution(
            torch.tensor(relative_noise_stddev, device=self._device)
        )

    def _log_prob(self, parameters: Tensor) -> LogProb:
        flattened_errors = self._calculate_flattened_errors(parameters)
        return self._error_distribution.log_prob(flattened_errors)


class LikelihoodEstimatedNoise(LikelihoodBase):
    def __init__(
        self,
        model: ModelProtocol,
        min_noise_stddev: float,
        inputs: Tensor,
        test_cases: Tensor,
        outputs: Tensor,
        device: Device,
    ) -> None:
        super().__init__(
            model=model,
            inputs=inputs,
            test_cases=test_cases,
            outputs=outputs,
            min_noise_stddev=min_noise_stddev,
            device=device,
        )

    def _log_prob(self, parameters: Tensor) -> LogProb:
        relative_noise_stddev, model_parameters = self._split_parameters(parameters)
        error_distribution = self._create_error_distribution(relative_noise_stddev)
        flattened_errors = self._calculate_flattened_errors(model_parameters)
        return error_distribution.log_prob(flattened_errors)

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        relative_noise_stddev = unsqueeze_if_necessary(parameters[0])
        model_parameters = unsqueeze_if_necessary(parameters[1:])
        return relative_noise_stddev, model_parameters


def create_likelihood(
    model: ModelProtocol,
    relative_noise_stddev: Optional[float],
    min_noise_stddev: float,
    inputs: Tensor,
    test_cases: Tensor,
    outputs: Tensor,
    device: Device,
) -> LikelihoodProtocol:
    if relative_noise_stddev is None:
        return LikelihoodEstimatedNoise(
            model=model,
            min_noise_stddev=min_noise_stddev,
            inputs=inputs,
            test_cases=test_cases,
            outputs=outputs,
            device=device,
        )
    else:
        return LikelihoodFixedNoise(
            model=model,
            relative_noise_stddev=relative_noise_stddev,
            min_noise_stddev=min_noise_stddev,
            inputs=inputs,
            test_cases=test_cases,
            outputs=outputs,
            device=device,
        )
