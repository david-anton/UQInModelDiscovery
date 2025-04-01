from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from torch import vmap

from bayesianmdisc.bayes.utility import flatten_tensor, repeat_tensor
from bayesianmdisc.errors import LikelihoodError
from bayesianmdisc.models import Model
from bayesianmdisc.statistics.distributions import (
    IndependentMultivariateNormalDistribution,
    create_independent_multivariate_normal_distribution,
)
from bayesianmdisc.customtypes import Device, Tensor

Prob: TypeAlias = Tensor
LogProb: TypeAlias = Tensor
GradLogProb: TypeAlias = Tensor
MSE: TypeAlias = Tensor


class LikelihoodProtocol(Protocol):
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


def calculate_mean_squared_error(y_predicted: Tensor, y_true: Tensor) -> MSE:
    return torch.mean(torch.square(calculate_error(y_predicted, y_true)))


class ErrorCalculator:
    def calculate(self, y_predicted: Tensor, y_true: Tensor) -> Tensor:
        return calculate_error(y_predicted, y_true)

    def calculate_mse(self, y_predicted: Tensor, y_true: Tensor) -> Tensor:
        return calculate_mean_squared_error(y_predicted, y_true)


class ErrorDistributionCreator:
    def __init__(self, device: Device) -> None:
        self._device = device

    def create(
        self, noise_stddevs: Tensor, num_outputs: int
    ) -> IndependentMultivariateNormalDistribution:
        means = self._assemble_means(noise_stddevs, num_outputs)
        stddevs = self._assemble_standard_deviations(noise_stddevs, num_outputs)
        return create_independent_multivariate_normal_distribution(
            means, stddevs, self._device
        )

    def _assemble_means(self, noise_stddevs: Tensor, num_outputs: int) -> Tensor:
        dim = self._calculate_dimension(noise_stddevs, num_outputs)
        return torch.zeros((dim,), device=self._device)

    def _assemble_standard_deviations(
        self, noise_stddevs: Tensor, num_outputs: int
    ) -> Tensor:
        dim = self._calculate_dimension(noise_stddevs, num_outputs)
        repeatet_stddevs = repeat_tensor(noise_stddevs, torch.Size([num_outputs, 1]))
        flattened_stddevs = flatten_tensor(repeatet_stddevs)
        return flattened_stddevs * torch.ones((dim,), device=self._device)

    def _calculate_dimension(
        self, single_output_noise: Tensor, num_outputs: int
    ) -> int:
        output_dim = len(single_output_noise)
        return num_outputs * output_dim


@dataclass
class MSELossStatistics:
    mean: float
    stddev: float


class Likelihood:
    def __init__(
        self,
        model: Model,
        noise_stddev: Tensor,
        inputs: Tensor,
        test_cases: Tensor,
        outputs: Tensor,
        device: Device,
    ) -> None:
        self._validate_data(inputs, outputs)
        self.noise_stddev = noise_stddev
        self._model = model
        self._inputs = inputs
        self._test_cases = test_cases
        self._true_outputs = outputs
        self._num_outputs = len(outputs)
        self._flattened_true_outputs = flatten_tensor(outputs)
        self._device = device
        self._error_calculator = ErrorCalculator()
        self._error_distribution_creator = ErrorDistributionCreator(self._device)
        self._error_distribution = self._error_distribution_creator.create(
            self.noise_stddev, self._num_outputs
        )

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

    def mse_loss_statistics(self, parameters: Tensor) -> MSELossStatistics:
        mean_squared_errors = self._calculate_mse_losses(parameters)
        mean = torch.mean(mean_squared_errors, dim=0).detach().cpu().item()
        stddev = torch.std(mean_squared_errors, dim=0).detach().cpu().item()
        return MSELossStatistics(
            mean=mean,
            stddev=stddev,
        )

    def _validate_data(self, inputs: Tensor, outputs: Tensor) -> None:
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        if not num_inputs == num_outputs:
            raise LikelihoodError(
                f"""The number of inputs and outputs is expected to be the same, 
                                  but is {num_inputs} and {num_outputs}."""
            )

    def _prob(self, parameters: Tensor) -> Prob:
        return torch.exp(self._log_prob(parameters))

    def _log_prob(self, parameters: Tensor) -> LogProb:
        flattened_errors = self._calculate_flattened_errors(parameters)
        return self._error_distribution.log_prob(flattened_errors)

    def _calculate_flattened_errors(self, parameters: Tensor) -> Tensor:
        outputs = self._model(self._inputs, self._test_cases, parameters)
        flattened_outputs = flatten_tensor(outputs)
        return self._error_calculator.calculate(
            flattened_outputs, self._flattened_true_outputs
        )

    def _calculate_mse_losses(self, parameters: Tensor) -> MSE:

        def vmap_mse_losses(parameters_: Tensor) -> MSE:
            outputs = self._model(self._inputs, self._test_cases, parameters_)
            flattened_outputs = flatten_tensor(outputs)
            return self._error_calculator.calculate_mse(
                flattened_outputs, self._flattened_true_outputs
            )

        mean_squared_errors = vmap(vmap_mse_losses)(parameters)
        return mean_squared_errors
