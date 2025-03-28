import math
from typing import Iterator, Protocol, TypeAlias

import torch
import torch.nn as nn

from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
    create_independent_multivariate_normal_distributed_prior,
    create_independent_multivariate_studentT_distributed_prior,
)
from bayesianmdisc.errors import GPPriorError
from bayesianmdisc.models import Model
from bayesianmdisc.types import Device, Parameter, Tensor

NumLayersList: TypeAlias = list[int]


class ParameterPrior(Protocol):
    def forward(self, num_samples: int) -> Tensor:
        pass

    def get_prior_distribution(self) -> PriorProtocol:
        pass

    def print_hyperparameters(self) -> None:
        pass

    def __call__(self, num_samples: int) -> Tensor:
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        pass


def create_parameter_prior(
    prior_type: str,
    is_mean_trainable: bool,
    model_library: Model,
    device: Device,
) -> ParameterPrior:
    if prior_type == "Gamma":
        if not is_mean_trainable:
            GPPriorError("Gamma prior has always a trainable mean.")
        return GammaParameterPrior(
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
        model: Model,
        device: Device,
    ):
        super().__init__()
        self._dim = model.num_parameters
        self._device = device
        initial_rho_shape = math.log(math.exp(1.0) - 1.0)
        initial_rho_rate = math.log(math.exp(1.0) - 1.0)
        self._rhos_shapes = self._init_rhos(initial_rho_shape)
        self._rhos_rates = self._init_rhos(initial_rho_rate)

    def forward(self, num_samples: int) -> Tensor:
        # shape = concentrations (PyTorch)
        shapes, rates = self._shapes_and_rates()
        return torch.distributions.Gamma(concentration=shapes, rate=rates).rsample(
            torch.Size([num_samples])
        )

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

    def _init_rhos(self, initial_rho: float) -> Tensor:
        rhos = torch.full(
            (self._dim,),
            initial_rho,
            requires_grad=True,
            device=self._device,
        )
        return nn.Parameter(rhos)

    def _shapes_and_rates(self) -> tuple[Tensor, Tensor]:
        def shapes_and_rates_func(rhos: Tensor) -> Tensor:
            return torch.log(torch.tensor(1.0, device=self._device) + torch.exp(rhos))

        shapes = shapes_and_rates_func(self._rhos_shapes)
        rates = shapes_and_rates_func(self._rhos_rates)
        return shapes, rates


class GaussianMean(nn.Module):
    def __init__(
        self,
        model: Model,
        is_trainable: bool,
        device: Device,
    ) -> None:
        super().__init__()
        self._dim = model.num_parameters
        self._is_trainable = is_trainable
        self._device = device
        self._initial_mean = 0.0
        self._means = self._init_means()

    def forward(self) -> Tensor:
        return self._means

    def print_hyperparameters(self) -> None:
        print(f"Means: {self.forward().data.detach()}")

    def _init_means(self) -> Tensor:
        means = torch.full(
            (self._dim,),
            self._initial_mean,
            requires_grad=self._is_trainable,
            device=self._device,
        )
        return nn.Parameter(means).requires_grad_(self._is_trainable)


class GaussianParameterPrior(nn.Module):
    def __init__(
        self,
        model: Model,
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
        self._initial_rho = math.log(math.exp(1.0) - 1.0)
        self._rhos = self._init_rhos()

    def forward(self, num_samples: int) -> Tensor:
        sigmas = self._sigmas()
        means = self._means()
        return torch.distributions.Normal(loc=means, scale=sigmas).rsample(
            torch.Size([num_samples])
        )

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
        rhos = torch.full(
            (self._dim,),
            self._initial_rho,
            requires_grad=True,
            device=self._device,
        )
        return nn.Parameter(rhos)

    def _sigmas(self) -> Tensor:
        def sigmas_func(rhos: Tensor) -> Tensor:
            return torch.log(torch.tensor(1.0, device=self._device) + torch.exp(rhos))

        return sigmas_func(self._rhos)


class HierarchicalGaussianParameterPrior(nn.Module):
    def __init__(
        self,
        model: Model,
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
        self._initial_rho_shape = math.log(math.exp(1.0) - 1.0)
        self._initial_rho_rate = math.log(math.exp(1.0) - 1.0)
        self._rhos_shapes = self._init_rhos_shapes()
        self._rhos_rates = self._init_rhos_rates()

    def forward(self, num_samples: int) -> Tensor:
        sigmas = self._sigmas()
        means = self._means()
        return torch.distributions.Normal(loc=means, scale=sigmas).rsample(
            torch.Size([num_samples])
        )

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

    def _init_rhos_shapes(self) -> Tensor:
        rhos_shapes = torch.full(
            (self._dim,),
            self._initial_rho_shape,
            requires_grad=True,
            device=self._device,
        )
        return nn.Parameter(rhos_shapes)

    def _init_rhos_rates(self) -> Tensor:
        rhos_rates = torch.full(
            (self._dim,),
            self._initial_rho_rate,
            requires_grad=True,
            device=self._device,
        )
        return nn.Parameter(rhos_rates)

    def _sigmas(self) -> Tensor:
        shapes, rates = self._shapes_and_rates()
        variances = torch.distributions.InverseGamma(
            concentration=shapes, rate=rates
        ).rsample()
        return torch.sqrt(variances)

    def _shapes_and_rates(self) -> tuple[Tensor, Tensor]:
        def shapes_and_rates_func(rhos: Tensor) -> Tensor:
            return torch.log(torch.tensor(1.0, device=self._device) + torch.exp(rhos))

        rhos_shapes = self._rhos_shapes
        rhos_rates = self._rhos_rates
        shapes = shapes_and_rates_func(rhos_shapes)
        rates = shapes_and_rates_func(rhos_rates)
        return shapes, rates
