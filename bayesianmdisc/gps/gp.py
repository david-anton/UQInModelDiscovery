from typing import Optional

import gpytorch
import torch

from bayesianmdisc.errors import GPError
from bayesianmdisc.gps.base import (
    GPMultivariateNormal,
    NamedParameters,
    TrainingDataTuple,
    validate_likelihood_noise_variance,
    validate_training_data,
)
from bayesianmdisc.gps.kernels import Kernel, ScaledMaternKernel, ScaledRBFKernel
from bayesianmdisc.gps.means import ZeroMean
from bayesianmdisc.gps.normalizers import InputNormalizer
from bayesianmdisc.gps.utility import validate_parameters_size
from bayesianmdisc.customtypes import Device, Tensor


class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        mean: ZeroMean,
        kernel: Kernel,
        input_normalizer: InputNormalizer,
        device: Device,
        train_x: TrainingDataTuple = (None,),
        train_y: TrainingDataTuple = (None,),
    ) -> None:
        inputs, targets = self._preprocess_training_data(train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Positive()
        ).to(device)
        super().__init__(inputs, targets, likelihood)
        self._mean = mean
        self._kernel = kernel
        self._input_normalizer = input_normalizer
        self.num_gps = 1
        self.num_hyperparameters = mean.num_hyperparameters + kernel.num_hyperparameters
        self._device = device

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        norm_x = self._input_normalizer(x)
        mean = self._mean(norm_x)
        covariance_matrix = self._kernel(norm_x, norm_x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance_matrix)

    def forward_mean(self, x: Tensor) -> Tensor:
        norm_x = self._input_normalizer(x)
        return self._mean(norm_x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        norm_x_1 = self._input_normalizer(x_1)
        norm_x_2 = self._input_normalizer(x_2)
        lazy_covariance_matrix = self._kernel(norm_x_1, norm_x_2)
        return lazy_covariance_matrix.to_dense()

    def forward_for_optimization(self, x: Tensor) -> GPMultivariateNormal:
        return self.forward(x)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        self._kernel.set_parameters(parameters)

    def get_named_parameters(self) -> NamedParameters:
        return self._kernel.get_named_parameters()

    # @override
    def set_train_data(
        self,
        inputs: TrainingDataTuple,
        targets: TrainingDataTuple,
        strict: bool = True,
    ) -> None:
        validate_training_data(inputs, targets, self.num_gps)
        _inputs, _targets = self._preprocess_training_data(inputs, targets)
        self._set_training_data(_inputs, _targets, strict)

    def add_train_data(
        self, inputs: TrainingDataTuple, targets: TrainingDataTuple
    ) -> None:
        validate_training_data(inputs, targets, self.num_gps)
        _inputs, _targets = self._preprocess_training_data(inputs, targets)
        all_inputs = self._append_inputs(_inputs)
        all_targets = self._append_targets(_targets)
        self._set_training_data(all_inputs, all_targets, strict=False)

    def set_likelihood_noise_variance(
        self, noise_variance: Tensor, is_trainable: bool
    ) -> None:
        validate_likelihood_noise_variance(noise_variance, self.num_gps)
        self.likelihood.noise_covar.noise = noise_variance
        self.likelihood.noise_covar.raw_noise.requires_grad_(is_trainable)

    def get_likelihood_noise_variance(self) -> Tensor:
        return self.likelihood.noise_covar.noise

    # @override
    def get_fantasy_model(
        self, inputs: TrainingDataTuple, targets: TrainingDataTuple, **kwargs
    ):
        validate_training_data(inputs, targets, self.num_gps)
        _inputs, _targets = self._preprocess_training_data(inputs, targets)
        return super().get_fantasy_model(_inputs, _targets, **kwargs)

    def _preprocess_training_data(
        self, train_x: TrainingDataTuple, train_y: TrainingDataTuple
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        inputs, targets = train_x[0], train_y[0]
        inputs, targets = self._copy_training_data_to_device(inputs, targets)
        targets = self._adjust_trainig_data_output_shape(targets)
        return inputs, targets

    def _copy_training_data_to_device(
        self, train_x: Optional[Tensor], train_y: Optional[Tensor]
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if train_x is not None and train_y is not None:
            train_x.to(self._device)
            train_y.to(self._device)
        return train_x, train_y

    def _adjust_trainig_data_output_shape(
        self, train_y: Optional[Tensor]
    ) -> Optional[Tensor]:
        if train_y is not None:
            if train_y.dim() == 2:
                train_y = torch.squeeze(train_y, dim=1)
        return train_y

    def _append_inputs(self, inputs: Optional[Tensor]) -> Optional[Tensor]:
        old_inputs = self.train_inputs
        if old_inputs is not None and inputs is not None:
            all_inputs = torch.concatenate((old_inputs[0], inputs), dim=0)
        else:
            all_inputs = inputs
        return all_inputs

    def _append_targets(self, targets: Optional[Tensor]) -> Optional[Tensor]:
        old_targets = self.train_targets
        if old_targets is not None and targets is not None:
            all_targets = torch.concatenate((old_targets, targets), dim=0)
        else:
            all_targets = targets
        return all_targets

    def _set_training_data(
        self,
        normalized_processed_inputs: Optional[Tensor],
        processed_targets: Optional[Tensor],
        strict: bool,
    ) -> None:
        if normalized_processed_inputs == None and processed_targets == None:
            self.train_inputs = None
            self.train_targets = None
        else:
            super().set_train_data(
                normalized_processed_inputs, processed_targets, strict
            )


def create_scaled_rbf_gaussian_process(
    mean: str,
    input_dims: int,
    min_inputs: Tensor,
    max_inputs: Tensor,
    device: Device,
    jitter: float = 0.0,
    train_x: TrainingDataTuple = (None,),
    train_y: TrainingDataTuple = (None,),
) -> GP:
    mean_module = _create_mean(mean, device)
    kernel_module = ScaledRBFKernel(input_dims, jitter, device).to(device)
    input_normalizer = _create_input_normalizer(min_inputs, max_inputs, device)
    return GP(
        mean=mean_module,
        kernel=kernel_module,
        input_normalizer=input_normalizer,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def create_scaled_matern_gaussian_process(
    mean: str,
    smoothness_parameter: float,
    input_dims: int,
    min_inputs: Tensor,
    max_inputs: Tensor,
    device: Device,
    jitter: float = 0.0,
    train_x: TrainingDataTuple = (None,),
    train_y: TrainingDataTuple = (None,),
) -> GP:
    mean_module = _create_mean(mean, device)
    kernel_module = ScaledMaternKernel(
        smoothness_parameter, input_dims, jitter, device
    ).to(device)
    input_normalizer = _create_input_normalizer(min_inputs, max_inputs, device)
    return GP(
        mean=mean_module,
        kernel=kernel_module,
        input_normalizer=input_normalizer,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def _create_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor, device: Device
) -> InputNormalizer:
    return InputNormalizer(min_inputs, max_inputs, device).to(device)


def _create_mean(mean: str, device: Device) -> ZeroMean:
    if mean == "zero":
        return ZeroMean(device).to(device)
    else:
        raise GPError(
            f"""There is no implementation for the requested 
            Gaussian process mean: {mean}."""
        )
