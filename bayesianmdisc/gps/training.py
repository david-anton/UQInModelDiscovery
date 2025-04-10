from typing import TypeAlias, cast

import gpytorch
import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.errors import GPError
from bayesianmdisc.gps.base import MarginalLogLikelihood
from bayesianmdisc.gps.gp import GP
from bayesianmdisc.gps.multioutputgp import IndependentMultiOutputGP
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_statistical_loss_history,
)

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP

print_interval = 10


def optimize_gp_hyperparameters(
    gaussian_process: GaussianProcess,
    inputs: Tensor,
    outputs: Tensor,
    initial_noise_stddevs: Tensor,
    num_iterations: int,
    learning_rate: float,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    _set_noise_standard_deviations(
        gaussian_process, inputs, initial_noise_stddevs, device
    )

    print("Start optimization of gaussian process hyperparameters ...")
    inputs, outputs = _preprocess_training_data(inputs, outputs, device)

    gaussian_process.eval()

    _set_training_data(gaussian_process, inputs, outputs)
    gaussian_process.train()
    gaussian_process.likelihood.train()

    optimizer = torch.optim.Adam(
        params=gaussian_process.parameters(),
        lr=learning_rate,
    )

    gp_outputs = _get_reshaped_gp_outputs(gaussian_process)
    marginal_log_likelihood = _create_marginal_log_likelihood(gaussian_process)

    def loss_func() -> Tensor:
        output_dustribution = gaussian_process.forward_for_optimization(inputs)
        return -marginal_log_likelihood(output_dustribution, gp_outputs)

    def loss_func_closure() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_func()
        loss.backward(retain_graph=True)
        return loss.item()

    def print_progress(iteration: int, loss: Tensor) -> None:
        if print_condition(iteration):
            print(f"negative mll: {loss.detach().cpu().item()}")

    def print_condition(iteration: int) -> bool:
        is_first = iteration == 1
        is_last = iteration == num_iterations
        is_interval_reached = iteration % print_interval == 0
        return is_first | is_last | is_interval_reached

    loss_hist = []
    for iteration in range(num_iterations):
        optimizer.step(loss_func_closure)
        loss = loss_func()
        print_progress(iteration, loss)
        loss_hist += [loss.detach().cpu().item()]

    _reset_training_data(gaussian_process)
    gaussian_process.eval()
    gaussian_process.likelihood.eval()

    # Postprocessing
    history_plotter_config = HistoryPlotterConfig()
    plot_statistical_loss_history(
        loss_hist=loss_hist,
        statistical_quantity="negative mll",
        file_name="loss_negative_mll_gp_training.png",
        output_subdir=output_subdirectory,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    print("Optimization finished.")
    print(
        f"""GP parameters: 
        {gaussian_process.get_named_parameters()}"""
    )
    print(
        f"""Noise standard deviation: 
        {_get_noise_standard_deviation(gaussian_process)}"""
    )


def condition_gp(
    gaussian_process: GaussianProcess,
    inputs: Tensor,
    outputs: Tensor,
    noise_stddevs: Tensor,
    device: Device,
) -> None:
    _set_noise_standard_deviations(gaussian_process, inputs, noise_stddevs, device)
    inputs, outputs = _preprocess_training_data(inputs, outputs, device)
    _set_training_data(gaussian_process, inputs, outputs)


def _preprocess_training_data(
    inputs: Tensor, outputs: Tensor, device: Device
) -> tuple[Tensor, Tensor]:
    _validate_training_data(inputs, outputs)
    inputs = inputs.to(device).detach()
    outputs = outputs.to(device).detach()
    return inputs, outputs


def _validate_training_data(inputs: Tensor, outputs: Tensor) -> None:
    num_inputs = len(inputs)
    num_outputs = len(outputs)
    if num_inputs != num_outputs:
        raise GPError(
            f"""Number of inputs {num_inputs} is expected 
            to be the same as number of outputs {num_outputs}."""
        )


def _set_noise_standard_deviations(
    gaussian_process: GaussianProcess, inputs: Tensor, noise_stddevs, device: Device
) -> None:
    is_noise_standard_deviation_trainable = False
    num_inputs = len(inputs)
    num_noise_stddevs = len(noise_stddevs)
    is_noise_heteroscedastic = num_inputs == num_noise_stddevs

    if is_noise_heteroscedastic:
        print("Use heteroscedastic noise for hyperparameter optimization")

        _validate_heteroscedastic_noise_standard_deviations(
            noise_stddevs, gaussian_process
        )
        _set_heteroscedastic_noise_dtandard_deviations(
            noise_stddevs,
            gaussian_process,
            is_noise_standard_deviation_trainable,
            device,
        )

    else:
        _set_homoscedastic_noise_standard_deviation(
            noise_stddevs,
            gaussian_process,
            is_noise_standard_deviation_trainable,
        )


def _set_homoscedastic_noise_standard_deviation(
    noise_standard_deviations: Tensor,
    gaussian_process: GaussianProcess,
    is_noise_trainable: bool,
) -> None:
    noise_variances = _calculate_variance_from_std(noise_standard_deviations)
    gaussian_process.set_likelihood_noise_variance(noise_variances, is_noise_trainable)


def _validate_heteroscedastic_noise_standard_deviations(
    initial_noise_stddevs: Tensor, gaussian_process: GaussianProcess
) -> None:
    noise_dims = initial_noise_stddevs.shape[1]
    output_dims = gaussian_process.num_gps
    if not noise_dims == output_dims:
        raise GPError(
            f"""It is expected that the second dimension of the heteroscedastic 
                noise tensor corresponds to the output dimension of the GP, 
                but is {noise_dims} and {output_dims}."""
        )


def _set_heteroscedastic_noise_dtandard_deviations(
    noise_standard_deviations: Tensor,
    gaussian_process: GaussianProcess,
    is_noise_trainable: bool,
    device: Device,
) -> None:
    output_dims = noise_standard_deviations.shape[1]
    noise_variance = _calculate_variance_from_std(noise_standard_deviations)
    likelihood_list = tuple(
        gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise_variance[:, output_dim],
            learn_additional_noise=is_noise_trainable,
        ).to(device)
        for output_dim in range(output_dims)
    )
    gaussian_process.set_likelihood(likelihood_list)


def _calculate_variance_from_std(std: Tensor) -> Tensor:
    return torch.square(std)


def _set_training_data(
    gaussian_process: GaussianProcess, inputs: Tensor, outputs: Tensor
) -> None:
    num_outputs = gaussian_process.num_gps
    inputs_tuple = tuple([inputs for _ in range(num_outputs)])
    outputs_tuple = torch.split(outputs, split_size_or_sections=1, dim=1)
    gaussian_process.set_train_data(inputs_tuple, outputs_tuple, strict=False)


def _get_reshaped_gp_outputs(gaussian_process: GaussianProcess) -> Tensor:
    return cast(Tensor, gaussian_process.train_targets)


def _create_marginal_log_likelihood(
    gaussian_process: GaussianProcess,
) -> MarginalLogLikelihood:
    num_outputs = gaussian_process.num_gps
    likelihood = likelihood = gaussian_process.likelihood
    if num_outputs == 1:
        return gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=likelihood, model=gaussian_process
        )
    else:
        return gpytorch.mlls.SumMarginalLogLikelihood(
            likelihood=likelihood, model=gaussian_process.gps
        )


def _reset_training_data(gaussian_process: GaussianProcess) -> None:
    num_outputs = gaussian_process.num_gps
    inputs_and_outputs = tuple([None for _ in range(num_outputs)])
    gaussian_process.set_train_data(
        inputs_and_outputs, inputs_and_outputs, strict=False
    )


def _get_noise_standard_deviation(gaussian_process: GaussianProcess) -> Tensor:
    noise_variance = gaussian_process.get_likelihood_noise_variance()
    return _calculate_std_from_variance(noise_variance)


def _calculate_std_from_variance(variance: Tensor) -> Tensor:
    return torch.sqrt(variance)
