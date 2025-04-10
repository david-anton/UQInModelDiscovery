from typing import Iterator, Protocol, TypeAlias, cast, Dict, Any

import normflows as nf
import torch
import torch.nn as nn

from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
    create_independent_multivariate_half_normal_distributed_prior,
    create_independent_multivariate_inverse_gamma_distributed_prior,
    create_independent_multivariate_normal_distributed_prior,
    create_independent_multivariate_studentT_distributed_prior,
)
from bayesianmdisc.customtypes import Device, Module, NFFlow, Parameter, Tensor
from bayesianmdisc.errors import GPPriorError
from bayesianmdisc.models import ModelProtocol
from bayesianmdisc.normalizingflows.base import BaseDistributionProtocol
from bayesianmdisc.normalizingflows.flows import (
    NormalizingFlow,
    create_exponential_constrained_flow,
    create_real_nvp_flow,
)
from bayesianmdisc.normalizingflows.prior import NormalizingFlowPrior
from bayesianmdisc.normalizingflows.utility import freeze_model

NumLayersList: TypeAlias = list[int]
ParameterOptions: TypeAlias = list[Dict[str, Any]]


class ParameterPrior(Protocol):
    def __call__(self, num_samples: int) -> Tensor:
        pass

    def forward(self, num_samples: int) -> Tensor:
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        pass

    def get_parameters_and_options(self) -> ParameterOptions:
        pass

    def get_prior_distribution(self) -> PriorProtocol:
        pass

    def print_hyperparameters(self) -> None:
        pass


def create_parameter_prior(
    prior_type: str,
    is_mean_trainable: bool,
    model_library: ModelProtocol,
    device: Device,
) -> ParameterPrior:
    if prior_type == "Gamma":
        if not is_mean_trainable:
            GPPriorError("Gamma prior has always a trainable mean.")
        return GammaParameterPrior(
            model=model_library,
            device=device,
        )
    elif prior_type == "inverse Gamma":
        if not is_mean_trainable:
            GPPriorError("Inverse Gamma prior has always a trainable mean.")
        return InverseGammaParameterPrior(
            model=model_library,
            device=device,
        )
    elif prior_type == "half Gaussian":
        if is_mean_trainable:
            GPPriorError("Half Gaussian prior has never a trainable mean.")
        return HalfNormalParameterPrior(
            model=model_library,
            device=device,
        )
    elif prior_type == "normalizing flow":
        if not is_mean_trainable:
            GPPriorError("Gamma prior has always a trainable mean.")
        return NormalizingFlowParameterPrior(
            model=model_library,
            device=device,
        )
    elif prior_type == "Gaussian":
        return GaussianParameterPrior(
            model=model_library,
            is_mean_trainable=is_mean_trainable,
            device=device,
        )
    elif prior_type == "hierarchical Gaussian":
        return HierarchicalGaussianParameterPrior(
            model=model_library,
            is_mean_trainable=is_mean_trainable,
            device=device,
        )
    else:
        raise GPPriorError(
            f"There is no implementation for the requested prior type {prior_type}"
        )


class GammaParameterPrior(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        initial_shape = torch.tensor(0.1)
        initial_rate = torch.tensor(10.0)
        min_shape = torch.tensor(1e-6)
        max_shape = torch.tensor(1e4)
        min_rate = torch.tensor(1e-6)
        max_rate = torch.tensor(1e4)
        self._rhos_shapes = self._init_rhos(initial_shape, min_shape, max_shape)
        self._rhos_rates = self._init_rhos(initial_rate, min_rate, max_rate)
        self._learning_rate_shapes = 0.001
        self._learning_rates_rates = 0.01

    def forward(self, num_samples: int) -> Tensor:
        # shape = concentrations (PyTorch)
        shapes, rates = self._shapes_and_rates()
        return torch.distributions.Gamma(concentration=shapes, rate=rates).rsample(
            torch.Size([num_samples])
        )

    def get_parameters_and_options(self) -> ParameterOptions:
        return [
            {"params": self._rhos_shapes, "lr": self._learning_rate_shapes},
            {"params": self._rhos_rates, "lr": self._learning_rates_rates},
        ]

    def get_prior_distribution(self) -> PriorProtocol:
        # shape = concentrations (PyTorch)
        shapes, rates = self._shapes_and_rates()
        return create_independent_multivariate_gamma_distributed_prior(
            concentrations=shapes,
            rates=rates,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        _shapes, _rates = self._shapes_and_rates()
        shapes = _shapes.data.detach()
        rates = _rates.data.detach()
        print(f"Shapes: {shapes}")
        print(f"Rates: {rates}")

    def _init_rhos(
        self, initial_parameter: Tensor, min_parameter: Tensor, max_parameter: Tensor
    ) -> Tensor:
        rhos = (
            parameters_to_rhos(initial_parameter)
            .repeat((self._dim,))
            .requires_grad_(True)
            .to(self._device)
            .clamp(
                min=(parameters_to_rhos(min_parameter)),
                max=parameters_to_rhos(max_parameter),
            )
        )
        return nn.Parameter(rhos)

    def _shapes_and_rates(self) -> tuple[Tensor, Tensor]:
        shapes = rhos_to_parameters(self._rhos_shapes)
        rates = rhos_to_parameters(self._rhos_rates)
        return shapes, rates


class InverseGammaParameterPrior(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        initial_shape = torch.tensor(100.0)
        initial_rate = torch.tensor(0.01)
        min_shape = torch.tensor(1e-6)
        max_shape = torch.tensor(500)
        min_rate = torch.tensor(1e-6)
        max_rate = torch.tensor(500)
        self._rhos_shapes = self._init_rhos(initial_shape, min_shape, max_shape)
        self._rhos_rates = self._init_rhos(initial_rate, min_rate, max_rate)
        self._learning_rate_shapes = 0.1
        self._learning_rates_rates = 0.001

    def forward(self, num_samples: int) -> Tensor:
        # shape = concentrations (PyTorch)
        shapes, rates = self._shapes_and_rates()
        return torch.distributions.InverseGamma(
            concentration=shapes, rate=rates
        ).rsample(torch.Size([num_samples]))

    def get_parameters_and_options(self) -> ParameterOptions:
        return [
            {"params": self._rhos_shapes, "lr": self._learning_rate_shapes},
            {"params": self._rhos_rates, "lr": self._learning_rates_rates},
        ]

    def get_prior_distribution(self) -> PriorProtocol:
        # shape = concentrations (PyTorch)
        shapes, rates = self._shapes_and_rates()
        return create_independent_multivariate_inverse_gamma_distributed_prior(
            concentrations=shapes,
            rates=rates,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        _shapes, _rates = self._shapes_and_rates()
        shapes = _shapes.data.detach()
        rates = _rates.data.detach()
        print(f"Shapes: {shapes}")
        print(f"Rates: {rates}")

    def _init_rhos(
        self, initial_parameter: Tensor, min_parameter: Tensor, max_parameter: Tensor
    ) -> Tensor:
        rhos = (
            parameters_to_rhos(initial_parameter)
            .repeat((self._dim,))
            .requires_grad_(True)
            .to(self._device)
            .clamp(
                min=(parameters_to_rhos(min_parameter)),
                max=parameters_to_rhos(max_parameter),
            )
        )
        return nn.Parameter(rhos)

    def _shapes_and_rates(self) -> tuple[Tensor, Tensor]:
        shapes = rhos_to_parameters(self._rhos_shapes)
        rates = rhos_to_parameters(self._rhos_rates)
        return shapes, rates


class HalfNormalParameterPrior(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        self._initial_stddev = torch.tensor(0.01)
        self._rhos = self._init_rhos()
        self._learning_rate_rhos = 0.001

    def forward(self, num_samples: int) -> Tensor:
        standard_deviations = self._sigmas()
        return torch.distributions.HalfNormal(scale=standard_deviations).rsample(
            torch.Size([num_samples])
        )

    def get_parameters_and_options(self) -> ParameterOptions:
        return [{"params": self._rhos, "lr": self._learning_rate_rhos}]

    def get_prior_distribution(self) -> PriorProtocol:
        standard_deviations = self._sigmas().data.detach()
        return create_independent_multivariate_half_normal_distributed_prior(
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        standard_deviations = self._sigmas().data.detach()
        print(f"Standard deviations: {standard_deviations}")

    def _init_rhos(self) -> Tensor:
        rhos = (
            parameters_to_rhos(self._initial_stddev)
            .repeat((self._dim,))
            .requires_grad_(True)
            .to(self._device)
        )
        return nn.Parameter(rhos)

    def _sigmas(self) -> Tensor:
        return rhos_to_parameters(self._rhos)


class GaussianMean(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        is_trainable: bool,
        device: Device,
    ) -> None:
        super().__init__()
        self._dim = model.num_parameters
        self._is_trainable = is_trainable
        self._device = device
        self._initial_mean = torch.tensor(0.0)
        self._means = self._init_means()
        self._learning_rate_means = 0.001

    def forward(self) -> Tensor:
        return self._means

    def get_parameters_and_options(self) -> ParameterOptions:
        return [{"params": self._means, "lr": self._learning_rate_means}]

    def print_hyperparameters(self) -> None:
        print(f"Means: {self.forward().data.detach()}")

    def _init_means(self) -> Tensor:
        means = (
            self._initial_mean.repeat((self._dim,))
            .requires_grad_(self._is_trainable)
            .to(self._device)
        )
        return nn.Parameter(means).requires_grad_(self._is_trainable)


class GaussianParameterPrior(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        is_mean_trainable: bool,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        self._means = GaussianMean(
            model=model,
            is_trainable=is_mean_trainable,
            device=self._device,
        )
        self._initial_stddev = torch.tensor(1.0)
        self._rhos = self._init_rhos()
        self._learning_rate_rhos = 0.001

    def forward(self, num_samples: int) -> Tensor:
        standard_deviations = self._sigmas()
        means = self._means()
        return torch.distributions.Normal(loc=means, scale=standard_deviations).rsample(
            torch.Size([num_samples])
        )

    def get_parameters_and_options(self) -> ParameterOptions:
        mean_parameter_options = self._means.get_parameters_and_options()
        parameter_options = [{"params": self._rhos, "lr": self._learning_rate_rhos}]
        return mean_parameter_options + parameter_options

    def get_prior_distribution(self) -> PriorProtocol:
        means = self._means()
        standard_deviations = self._sigmas().data.detach()
        return create_independent_multivariate_normal_distributed_prior(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        self._means.print_hyperparameters()
        standard_deviations = self._sigmas().data.detach()
        print(f"Standard deviations: {standard_deviations}")

    def _init_rhos(self) -> Tensor:
        rhos = (
            parameters_to_rhos(self._initial_stddev)
            .repeat((self._dim,))
            .requires_grad_(True)
            .to(self._device)
        )
        return nn.Parameter(rhos)

    def _sigmas(self) -> Tensor:
        return rhos_to_parameters(self._rhos)


class HierarchicalGaussianParameterPrior(nn.Module):
    def __init__(
        self,
        model: ModelProtocol,
        is_mean_trainable: bool,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        self._means = GaussianMean(
            model=model,
            is_trainable=is_mean_trainable,
            device=self._device,
        )
        initial_shape = torch.tensor(1.0)
        initial_rate = torch.tensor(1.0)
        min_shape = torch.tensor(1e-6)
        max_shape = torch.tensor(1e4)
        min_rate = torch.tensor(1e-6)
        max_rate = torch.tensor(1e4)
        self._rhos_shapes = self._init_rhos(initial_shape, min_shape, max_shape)
        self._rhos_rates = self._init_rhos(initial_rate, min_rate, max_rate)
        self._learning_rate_shapes = 0.001
        self._learning_rates_rates = 0.001

    def forward(self, num_samples: int) -> Tensor:
        sigmas = self._sigmas()
        means = self._means()
        return torch.distributions.Normal(loc=means, scale=sigmas).rsample(
            torch.Size([num_samples])
        )

    def get_parameters_and_options(self) -> ParameterOptions:
        return [
            {"params": self._rhos_shapes, "lr": self._learning_rate_shapes},
            {"params": self._rhos_rates, "lr": self._learning_rates_rates},
        ]

    def get_prior_distribution(self) -> PriorProtocol:
        # alpha = shape
        # beta = scale
        shapes, rates = self._shapes_and_rates()
        means = self._means()
        degrees_of_freedom = 2 * shapes  # 2 * alpha
        scales = 1 / rates
        scales_studentT = scales / shapes  # beta / alpha
        return create_independent_multivariate_studentT_distributed_prior(
            degrees_of_freedom=degrees_of_freedom,
            means=means,
            scales=scales_studentT,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        self._means.print_hyperparameters()
        _shapes, _rates = self._shapes_and_rates()
        shapes = _shapes.data.detach()
        rates = _rates.data.detach()
        print(f"Shapes: {shapes}")
        print(f"Rates: {rates}")

    def _init_rhos(
        self, initial_parameter: Tensor, min_parameter: Tensor, max_parameter: Tensor
    ) -> Tensor:
        rhos = (
            parameters_to_rhos(initial_parameter)
            .repeat((self._dim,))
            .requires_grad_(True)
            .to(self._device)
            .clamp(
                min=(parameters_to_rhos(min_parameter)),
                max=parameters_to_rhos(max_parameter),
            )
        )
        return nn.Parameter(rhos)

    def _sigmas(self) -> Tensor:
        shapes, rates = self._shapes_and_rates()
        variances = torch.distributions.InverseGamma(
            concentration=shapes, rate=rates
        ).rsample()
        return torch.sqrt(variances)

    def _shapes_and_rates(self) -> tuple[Tensor, Tensor]:
        shapes = rhos_to_parameters(self._rhos_shapes)
        rates = rhos_to_parameters(self._rhos_rates)
        return shapes, rates


class NormalizingFlowParameterPrior(nn.Module):
    def __init__(self, model: ModelProtocol, device: Device) -> None:
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        self._is_base_trainable = False
        self._num_layers = 16
        self._relative_width_layers = 2.0
        self._normalizing_flow = self._init_normalizing_flow()

    def forward(self, num_samples: int) -> Tensor:
        samples, _ = self._normalizing_flow.sample(num_samples)
        return samples

    def get_parameters_and_options(self) -> ParameterOptions:
        return [{"params": self._normalizing_flow.parameters(), "lr": 1.0}]

    def get_prior_distribution(self) -> PriorProtocol:
        freeze_model(cast(Module, self._normalizing_flow))
        return NormalizingFlowPrior(
            normalizing_flow=self._normalizing_flow,
            dim=self._dim,
            device=self._device,
        )

    def print_hyperparameters(self) -> None:
        self._normalizing_flow.print_summary()

    def _init_normalizing_flow(self) -> NormalizingFlow:
        base_distribution = self._init_base_distribution()
        flows = self._init_normalizing_flows()
        return NormalizingFlow(flows, base_distribution, self._device).to(self._device)

    def _init_base_distribution(self) -> BaseDistributionProtocol:
        return nf.distributions.base.DiagGaussian(
            self._dim, trainable=self._is_base_trainable
        ).to(self._device)

    def _init_normalizing_flows(self) -> list[NFFlow]:
        width_layers = int(self._relative_width_layers * self._dim)
        indices_constrained_outputs = [_ for _ in range(self._dim)]
        flows: list[NFFlow] = [
            create_real_nvp_flow(
                number_inputs=self._dim,
                width_hidden_layer=width_layers,
            )
            for _ in range(self._num_layers)
        ]
        flows += [
            create_exponential_constrained_flow(
                total_num_outputs=self._dim,
                indices_constrained_outputs=indices_constrained_outputs,
            )
        ]
        return flows


def parameters_to_rhos(parameters: Tensor) -> Tensor:
    return torch.log(torch.exp(parameters) - 1.0)


def rhos_to_parameters(rhos: Tensor) -> Tensor:
    return torch.log(1.0 + torch.exp(rhos))
