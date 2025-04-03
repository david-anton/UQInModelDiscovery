from itertools import groupby
from typing import TypeAlias

import gpytorch
import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.errors import GPError
from bayesianmdisc.gps.base import (
    GPMultivariateNormal,
    GPMultivariateNormalList,
    GPLikelihoodsTuple,
    NamedParameters,
    TrainingDataTuple,
    validate_likelihood_noise_variance,
    validate_training_data,
    validate_likelihoods,
)
from bayesianmdisc.gps.gp import GP
from bayesianmdisc.gps.utility import validate_parameters_size

GPTuple: TypeAlias = tuple[GP, ...]
GPList: TypeAlias = list[GP]
GPMultivariateNormalTuple: TypeAlias = tuple[GPMultivariateNormal]
GPIndependentGPList: TypeAlias = gpytorch.models.IndependentModelList
GPLikelihoodList: TypeAlias = gpytorch.likelihoods.LikelihoodList


class IndependentMultiOutputGP(gpytorch.models.GP):
    def __init__(
        self,
        gps: GPTuple,
        device: Device,
    ) -> None:
        super().__init__()
        self._device = device
        self.gps = self._prepare_gp_list(gps)
        self.likelihood = self._prepare_likelihood_list_from_gps(gps)
        self.num_gps = len(gps)
        self.num_hyperparameters = self._determine_number_of_hyperparameters(gps)

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        multivariate_normals = self.gps(*[x for _ in range(self.num_gps)])
        return _combine_independent_multivariate_normals(
            multivariate_normals, self._device
        )

    def forward_mean(self, x: Tensor) -> Tensor:
        means = [gp.forward_mean(x) for gp in self.gps.models]
        return torch.concat(means, dim=0).to(self._device)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        num_outputs = self.num_gps
        num_inputs_1 = x_1.size()[0]
        num_inputs_2 = x_2.size()[0]

        sub_covar_matrices = []
        sub_covar_matrices.append(
            torch.concat(
                (
                    self.gps.models[0].forward_kernel(x_1, x_2),
                    _create_zeros(
                        num_inputs_1, (num_outputs - 1) * num_inputs_2, self._device
                    ),
                ),
                dim=1,
            )
        )

        for i in range(1, num_outputs - 1):
            zeros_left = _create_zeros(num_inputs_1, i * num_inputs_2, self._device)
            sub_covar_matrix = self.gps.models[i].forward_kernel(x_1, x_2)
            zeros_right = _create_zeros(
                num_inputs_1, (num_outputs - 1 - i) * num_inputs_2, self._device
            )
            sub_covar_matrices.append(
                torch.concat((zeros_left, sub_covar_matrix, zeros_right), dim=1)
            )

        sub_covar_matrices.append(
            torch.concat(
                (
                    _create_zeros(
                        num_inputs_1, (num_outputs - 1) * num_inputs_2, self._device
                    ),
                    self.gps.models[-1].forward_kernel(x_1, x_2),
                ),
                dim=1,
            )
        )
        return torch.concat(sub_covar_matrices, dim=0)

    def forward_for_optimization(self, x: Tensor) -> GPMultivariateNormalTuple:
        return self.gps(*[x for _ in range(self.num_gps)])

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        start_index = 0
        for gp in self.gps.models:
            num_parameters = gp.num_hyperparameters
            gp.set_parameters(
                parameters[start_index : start_index + num_parameters].to(self._device)
            )
            start_index += num_parameters

    def get_named_parameters(self) -> NamedParameters:
        return {
            f"{key}_dim_{count}": value
            for count, gp in enumerate(self.gps.models)
            for key, value in gp.get_named_parameters().items()
        }

    # @override
    def set_train_data(
        self,
        inputs: TrainingDataTuple,
        targets: TrainingDataTuple,
        strict: bool = True,
    ) -> None:
        validate_training_data(inputs, targets, self.num_gps)
        for i, gp in enumerate(self.gps.models):
            gp.set_train_data((inputs[i],), (targets[i],), strict)

    def add_train_data(
        self, inputs: TrainingDataTuple, targets: TrainingDataTuple
    ) -> None:
        validate_training_data(inputs, targets, self.num_gps)
        for i, gp in enumerate(self.gps.models):
            gp.add_train_data((inputs[i],), (targets[i],))

    def set_likelihood_noise_variance(
        self, noise_variance: Tensor, is_trainable: bool
    ) -> None:
        validate_likelihood_noise_variance(noise_variance, self.num_gps)
        for i, likelihood in enumerate(self.likelihood.likelihoods):
            likelihood.noise_covar.noise = torch.unsqueeze(noise_variance[i], dim=0)
            likelihood.noise_covar.raw_noise.requires_grad_(is_trainable)

    def get_likelihood_noise_variance(self) -> Tensor:
        noise_variances = [
            likelihood.noise_covar.noise for likelihood in self.likelihood.likelihoods
        ]
        noise_variances = [
            noise_variance.reshape((-1, 1)) for noise_variance in noise_variances
        ]
        return torch.concat(noise_variances, dim=1)

    def set_likelihood(self, likelihood: GPLikelihoodsTuple) -> None:
        validate_likelihoods(likelihood, self.num_gps)
        likelihood_list = self._prepare_likelihood_list(likelihood)
        self.likelihood = likelihood_list

    def _prepare_likelihood_list(
        self, likelihoods: GPLikelihoodsTuple
    ) -> GPLikelihoodList:
        return gpytorch.likelihoods.LikelihoodList(
            *[likelihood.to(self._device) for likelihood in likelihoods]
        )

    def _prepare_gp_list(self, gps: GPTuple) -> GPIndependentGPList:
        return gpytorch.models.IndependentModelList(
            *[gp.to(self._device) for gp in gps]
        )

    def _prepare_likelihood_list_from_gps(self, gps: GPTuple) -> GPLikelihoodList:
        return gpytorch.likelihoods.LikelihoodList(
            *[gp.likelihood.to(self._device) for gp in gps]
        )

    def _determine_number_of_hyperparameters(self, gps: GPTuple) -> int:
        return sum([gp.num_hyperparameters for gp in gps])

    @property
    def train_targets(self) -> Tensor:
        return self.gps.train_targets


def flatten_outputs(outputs: Tensor) -> Tensor:
    if outputs.dim() == 1:
        return outputs
    else:
        return torch.transpose(outputs, 1, 0).ravel()


def _combine_independent_multivariate_normals(
    multivariate_normals: GPMultivariateNormalList, device: Device
) -> GPMultivariateNormal:
    def _validate_equal_size(normals: GPMultivariateNormalList) -> None:
        normal_sizes = [normal.loc.size()[0] for normal in normals]
        grouped_sizes = groupby(normal_sizes)
        is_only_one_group = next(grouped_sizes, True) and not next(grouped_sizes, False)
        if not is_only_one_group:
            raise GPError(
                """It is expected that the independent multivariate normal distributions 
                are of equal size."""
            )

    def _combine_means(normals: GPMultivariateNormalList, device: Device) -> Tensor:
        means = tuple(normal.loc for normal in normals)
        return torch.concat(means, dim=0).to(device)

    def _combine_covariance_matrices(
        normals: GPMultivariateNormalList, device: Device
    ) -> Tensor:
        output_dims = _determine_number_of_output_dimensions(normals)
        output_size = _determine_output_size(normals)

        sub_covar_matrices = []
        sub_covar_matrices.append(
            torch.concat(
                (
                    normals[0].covariance_matrix,
                    _create_zeros(output_size, (output_dims - 1) * output_size, device),
                ),
                dim=1,
            )
        )

        for i in range(1, output_dims - 1):
            zeros_left = _create_zeros(output_size, i * output_size, device)
            sub_covar_matrix = normals[i].covariance_matrix
            zeros_right = _create_zeros(
                output_size, (output_dims - 1 - i) * output_size, device
            )
            sub_covar_matrices.append(
                torch.concat((zeros_left, sub_covar_matrix, zeros_right), dim=1)
            )

        sub_covar_matrices.append(
            torch.concat(
                (
                    _create_zeros(output_size, (output_dims - 1) * output_size, device),
                    normals[-1].covariance_matrix,
                ),
                dim=1,
            )
        )
        return torch.concat(sub_covar_matrices, dim=0).to(device)

    def _determine_number_of_output_dimensions(
        normals: GPMultivariateNormalList,
    ) -> int:
        return len(normals)

    def _determine_output_size(multivariate_normals: GPMultivariateNormalList) -> int:
        return multivariate_normals[0].loc.size()[0]

    normals = multivariate_normals
    _validate_equal_size(normals)
    combined_mean = _combine_means(normals, device)
    combined_covariance_matrix = _combine_covariance_matrices(normals, device)
    return gpytorch.distributions.MultivariateNormal(
        combined_mean, combined_covariance_matrix
    )


def _create_zeros(dim_1: int, dim_2: int, device: Device) -> Tensor:
    return torch.zeros(
        (dim_1, dim_2),
        requires_grad=True,
        device=device,
    )
