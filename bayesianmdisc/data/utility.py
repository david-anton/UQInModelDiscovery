import itertools
from dataclasses import dataclass
from typing import TypeAlias

import torch
from scipy.interpolate import NearestNDInterpolator

from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data.base import DeformationInputs, StressOutputs
from bayesianmdisc.datasettings import (
    data_set_label_anisotropic,
    data_set_label_kawabata,
    data_set_label_treloar,
)
from bayesianmdisc.errors import DataError
from bayesianmdisc.testcases import (
    TestCases,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.utility import from_numpy_to_torch, from_torch_to_numpy

NoiseStddevs: TypeAlias = Tensor


@dataclass
class SplittedData:
    inputs_prior: DeformationInputs
    inputs_posterior: DeformationInputs
    test_cases_prior: TestCases
    test_cases_posterior: TestCases
    outputs_prior: StressOutputs
    outputs_posterior: StressOutputs
    noise_stddevs_prior: Tensor
    noise_stddevs_posterior: Tensor


def split_data(
    data_set_label: str,
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> SplittedData:

    def split_treloar_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)
        mask_ut = torch.where(test_cases == test_case_identifier_uniaxial_tension)[0]
        num_points_ut = torch.numel(mask_ut)
        mask_ebt = torch.where(test_cases == test_case_identifier_equibiaxial_tension)[
            0
        ]
        num_points_ebt = torch.numel(mask_ebt)

        # Relative indices
        rel_indices_prior_ut = [2, 6, 10, 15, 20]
        rel_indices_prior_ebt = [2, 6, 11]
        rel_indices_prior_ps = [2, 5, 10]
        # Absolute indices
        indices_prior_ut = rel_indices_prior_ut
        start_index = num_points_ut
        indices_prior_ebt = [i + start_index for i in rel_indices_prior_ebt]
        start_index = num_points_ut + num_points_ebt
        indices_prior_ps = [i + start_index for i in rel_indices_prior_ps]
        indices_prior = indices_prior_ut + indices_prior_ebt + indices_prior_ps
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    def split_kawabata_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)

        # Indices
        indices_prior = [
            2,
            5,
            10,
            14,
            17,
            22,
            25,
            30,
            34,
            38,
            42,
            46,
            50,
            54,
            57,
            61,
            64,
            68,
            70,
        ]
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    def split_anisotropic_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)

        # Indices
        num_data_sets = 11
        num_points_per_dataset = 11
        indices_prior_lists = [
            [1 + i * num_points_per_dataset, 9 + i * num_points_per_dataset]
            for i in range(num_data_sets)
        ]
        indices_prior = list(itertools.chain.from_iterable(indices_prior_lists))
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    validate_data(inputs, test_cases, outputs, noise_stddevs)
    if data_set_label == data_set_label_treloar:
        return split_treloar_data(inputs, test_cases, outputs, noise_stddevs)
    elif data_set_label == data_set_label_kawabata:
        return split_kawabata_data(inputs, test_cases, outputs, noise_stddevs)
    elif data_set_label == data_set_label_anisotropic:
        return split_anisotropic_data(inputs, test_cases, outputs, noise_stddevs)
    else:
        raise DataError(
            f"No implementation for the requested data set {data_set_label}"
        )


def validate_data(
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> None:
    num_inputs = len(inputs)
    num_test_cases = len(test_cases)
    num_outputs = len(outputs)
    num_noise_stddevs = len(noise_stddevs)

    if (
        num_inputs != num_test_cases
        and num_inputs != num_outputs
        and num_inputs != num_noise_stddevs
    ):
        raise DataError(
            f"""The number of inputs, test cases, outputs and noise standard deviations 
                        is expected to be the same but is {num_inputs}, {num_test_cases}, 
                        {num_outputs} and {num_noise_stddevs}"""
        )


def determine_heteroscedastic_noise(
    relative_noise_stddevs: float | Tensor,
    min_absolute_noise_stddev: float,
    outputs: StressOutputs,
) -> NoiseStddevs:
    noise_stddevs = relative_noise_stddevs * outputs
    return torch.where(
        noise_stddevs < min_absolute_noise_stddev,
        min_absolute_noise_stddev,
        noise_stddevs,
    )


def interpolate_heteroscedastic_noise(
    new_inputs: DeformationInputs,
    new_test_cases: TestCases,
    inputs: DeformationInputs,
    test_cases: TestCases,
    noise_stddevs: NoiseStddevs,
    device: Device,
) -> NoiseStddevs:

    def validate_new_inputs(
        new_inputs: DeformationInputs, new_test_cases: TestCases
    ) -> None:
        num_new_inputs = len(new_inputs)
        num_new_test_cases = len(new_test_cases)

        if not num_new_inputs == num_new_test_cases:
            raise DataError(
                f"""The number of new inputs and new test cases is expected to be the same,
                but is {num_new_inputs} and {num_new_test_cases}."""
            )

    def validate_inputs(
        inputs: DeformationInputs, test_cases: TestCases, noise_stddevs: NoiseStddevs
    ) -> None:
        num_inputs = len(inputs)
        num_test_cases = len(test_cases)
        num_noise_stddevs = len(noise_stddevs)

        if not num_inputs == num_test_cases and num_inputs == num_noise_stddevs:
            raise DataError(
                f"""The number of inputs, test cases and noise standard deviations is expected to be the same,
                but is {num_inputs}, {num_test_cases} and {num_noise_stddevs}."""
            )

    def find_all_new_test_cases(new_test_cases: TestCases) -> list[int]:
        return list(set(new_test_cases.tolist()))

    def flatten_noise_stddevs(noise_stddevs: NPArray) -> NPArray:
        return noise_stddevs.reshape((-1,))

    def unflatten_noise_stddevs(noise_stddevs: NPArray) -> NPArray:
        return noise_stddevs.reshape((-1, 1))

    validate_new_inputs(new_inputs, new_test_cases)
    validate_inputs(inputs, test_cases, noise_stddevs)

    new_inputs_np = from_torch_to_numpy(new_inputs)
    new_test_cases_np = from_torch_to_numpy(new_test_cases)
    inputs_np = from_torch_to_numpy(inputs)
    test_cases_np = from_torch_to_numpy(test_cases)
    noise_stddevs_np = from_torch_to_numpy(noise_stddevs)

    output_dims = noise_stddevs.shape[1]
    new_test_cases_list = find_all_new_test_cases(new_test_cases)

    new_noise_stddevs_outputs = []

    for output_dim in range(output_dims):

        new_noise_stddevs_test_cases = []

        for new_test_case in new_test_cases_list:
            new_indices = new_test_case == new_test_cases_np
            indices = new_test_case == test_cases_np

            new_inputs_selected = new_inputs_np[new_indices]
            inputs_selected = inputs_np[indices]
            noise_stddevs_selected = noise_stddevs_np[indices, output_dim]
            noise_stddevs_selected = flatten_noise_stddevs(noise_stddevs_selected)

            interpolator = NearestNDInterpolator(
                inputs_selected, noise_stddevs_selected
            )
            new_noise_stddevs = interpolator(new_inputs_selected)
            new_noise_stddevs = unflatten_noise_stddevs(new_noise_stddevs)
            new_noise_stddevs_test_cases += [
                from_numpy_to_torch(new_noise_stddevs, device)
            ]

        new_noise_stddevs_outputs += [torch.concat(new_noise_stddevs_test_cases, dim=0)]

    return torch.concat(new_noise_stddevs_outputs, dim=1)


def add_noise_to_data(
    noise_stddevs: Tensor, outputs: StressOutputs, device: Device
) -> StressOutputs:

    def validate_inputs(noise_stddevs: Tensor, outputs: StressOutputs) -> None:
        size_noise_stddevs = noise_stddevs.size()
        size_outupts = outputs.size()

        if not size_noise_stddevs == size_outupts:
            raise DataError(
                f"""The standard deviations and outputs are expected to be of the same size
                but are {size_noise_stddevs} and {size_outupts}."""
            )

    validate_inputs(noise_stddevs, outputs)

    mean_noise = torch.zeros_like(outputs, device=device)
    noise = torch.normal(mean=mean_noise, std=noise_stddevs).to(device)
    return outputs + noise
