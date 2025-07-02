from typing import Optional, cast

import gpytorch
import torch
from torch import vmap

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.errors import GPError
from bayesianmdisc.gps.base import (
    GPLikelihoodsTuple,
    GPMultivariateNormal,
    NamedParameters,
    TrainingDataTuple,
    InputMask,
    validate_likelihood_noise_variance,
    validate_likelihoods,
    validate_training_data,
)
from bayesianmdisc.gps.kernels import Kernel, ScaledMaternKernel, ScaledRBFKernel
from bayesianmdisc.gps.means import LinearMean, NonZeroMean, ZeroMean
from bayesianmdisc.gps.normalizers import InputNormalizer
from bayesianmdisc.gps.utility import validate_parameters_size


class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        mean: ZeroMean | NonZeroMean,
        kernel: Kernel,
        input_normalizer: InputNormalizer,
        input_mask: Optional[InputMask],
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
        self._is_zero_mean = isinstance(self._mean, ZeroMean)
        self._input_normalizer = input_normalizer
        self._input_mask = input_mask
        self.num_gps = 1
        self.num_hyperparameters = mean.num_hyperparameters + kernel.num_hyperparameters
        self._device = device

    def __call__(self, x: Tensor) -> GPMultivariateNormal:
        _x = self._mask_inputs(x)
        return super().__call__(_x)

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        _x = self._preprocess_x(x)
        mean = self._mean(_x)
        covariance_matrix = self._kernel(_x, _x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance_matrix)

    def forward_mean(self, x: Tensor) -> Tensor:
        _x = self._preprocess_x(x)
        return self._mean(_x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        _x_1 = self._preprocess_x(x_1)
        _x_2 = self._preprocess_x(x_2)
        lazy_covariance_matrix = self._kernel(_x_1, _x_2)
        return lazy_covariance_matrix.to_dense()

    def forward_for_optimization(self, x: Tensor) -> GPMultivariateNormal:
        return self.forward(x)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        if self._is_zero_mean:
            self._kernel.set_parameters(parameters)
        else:
            self._mean = cast(NonZeroMean, self._mean)
            num_parameters_mean = self._mean.num_hyperparameters
            self._mean.set_parameters(parameters[:num_parameters_mean])
            self._kernel.set_parameters(parameters[num_parameters_mean:])

    def get_named_parameters(self) -> NamedParameters:
        parameters_kernel = self._kernel.get_named_parameters()
        if self._is_zero_mean:
            return parameters_kernel
        else:
            self._mean = cast(NonZeroMean, self._mean)
            parameters_mean = self._mean.get_named_parameters()
            return parameters_mean | parameters_kernel

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

    def set_likelihood(self, likelihood: GPLikelihoodsTuple) -> None:
        validate_likelihoods(likelihood, self.num_gps)
        self.likelihood = likelihood[0].to(self._device)

    def infer_predictive_distribution(
        self, x: Tensor, noise_stddevs: Optional[Tensor] = None
    ) -> GPMultivariateNormal:

        def validate_noise_stddevs(x: Tensor, noise_stddevs: Tensor) -> None:
            num_inputs = len(x)
            actual_size = noise_stddevs.shape
            expected_size = torch.Size([num_inputs, self.num_gps])

            if not actual_size == expected_size:
                raise GPError(
                    f"""The noise stdandard deviation has not the expected size
                         (actual size: {actual_size}, expected size{expected_size}) """
                )

        gp_distribution = self.__call__(x)
        likelihood = self.likelihood
        if noise_stddevs is not None:
            validate_noise_stddevs(x, noise_stddevs)
            noise_stddevs = noise_stddevs.reshape((-1,))
            noise_variance = noise_stddevs**2
            return likelihood(gp_distribution, noise=noise_variance)
        else:
            return likelihood(gp_distribution)

    def _preprocess_training_data(
        self, train_x: TrainingDataTuple, train_y: TrainingDataTuple
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        inputs, targets = train_x[0], train_y[0]
        if self._check_if_training_data(inputs, targets):
            inputs = self._copy_to_device(cast(Tensor, inputs))
            targets = self._copy_to_device(cast(Tensor, targets))
            inputs = self._mask_inputs(inputs)
            targets = self._adjust_target_shape(targets)
        return inputs, targets

    def _check_if_training_data(
        self, inputs: Optional[Tensor], targets: Optional[Tensor]
    ) -> bool:
        is_inputs = inputs is not None
        is_targets = targets is not None
        return is_inputs and is_targets

    def _copy_to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(self._device)

    def _adjust_target_shape(self, train_y: Tensor) -> Tensor:
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
        inputs: Optional[Tensor],
        targets: Optional[Tensor],
        strict: bool,
    ) -> None:
        if inputs == None and targets == None:
            self.train_inputs = None
            self.train_targets = None
        else:
            super().set_train_data(inputs, targets, strict)

    def _preprocess_x(self, inputs: Tensor) -> Tensor:
        masked_inputs = self._mask_inputs(inputs)
        return self._normalize_inputs(masked_inputs)

    def _normalize_inputs(self, inputs: Tensor) -> Tensor:
        return self._input_normalizer(inputs)

    def _mask_inputs(self, inputs: Tensor) -> Tensor:
        if self._input_mask is not None:
            mask_dim = self._input_mask.shape[0]
            input_dim = inputs.shape[1]
            if input_dim == mask_dim:
                vmap_func = lambda _inputs: _inputs[self._input_mask]
                return vmap(vmap_func)(inputs)
            else:
                return inputs
        else:
            return inputs


def create_scaled_rbf_gaussian_process(
    mean: str,
    input_dim: int,
    min_inputs: Tensor,
    max_inputs: Tensor,
    input_mask: Optional[InputMask],
    device: Device,
    jitter: float = 0.0,
    train_x: TrainingDataTuple = (None,),
    train_y: TrainingDataTuple = (None,),
) -> GP:
    mean_module = _create_mean(mean, input_dim, device)
    kernel_module = ScaledRBFKernel(input_dim, jitter, device).to(device)
    input_normalizer = _create_input_normalizer(
        min_inputs, max_inputs, input_mask, device
    )
    return GP(
        mean=mean_module,
        kernel=kernel_module,
        input_normalizer=input_normalizer,
        input_mask=input_mask,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def create_scaled_matern_gaussian_process(
    mean: str,
    smoothness_parameter: float,
    input_dim: int,
    min_inputs: Tensor,
    max_inputs: Tensor,
    input_mask: Optional[InputMask],
    device: Device,
    jitter: float = 0.0,
    train_x: TrainingDataTuple = (None,),
    train_y: TrainingDataTuple = (None,),
) -> GP:
    mean_module = _create_mean(mean, input_dim, device)
    kernel_module = ScaledMaternKernel(
        smoothness_parameter, input_dim, jitter, device
    ).to(device)
    input_normalizer = _create_input_normalizer(
        min_inputs, max_inputs, input_mask, device
    )
    return GP(
        mean=mean_module,
        kernel=kernel_module,
        input_normalizer=input_normalizer,
        input_mask=input_mask,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def _create_input_normalizer(
    min_inputs: Tensor,
    max_inputs: Tensor,
    input_mask: Optional[InputMask],
    device: Device,
) -> InputNormalizer:
    if input_mask is not None:
        min_inputs = min_inputs[input_mask]
        max_inputs = max_inputs[input_mask]
    return InputNormalizer(min_inputs, max_inputs, device).to(device)


def _create_mean(mean: str, input_dim: int, device: Device) -> ZeroMean | NonZeroMean:
    if mean == "zero":
        return ZeroMean(device).to(device)
    elif mean == "linear":
        return LinearMean(input_dim, device).to(device)
    else:
        raise GPError(
            f"""There is no implementation for the requested 
            Gaussian process mean: {mean}."""
        )
