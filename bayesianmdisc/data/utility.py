from dataclasses import dataclass

import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
    data_set_label_kawabata,
    data_set_label_treloar,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.errors import DataError, DataSetError


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

    validate_data(inputs, test_cases, outputs, noise_stddevs)
    if data_set_label == data_set_label_treloar:
        return split_treloar_data(inputs, test_cases, outputs, noise_stddevs)
    elif data_set_label == data_set_label_kawabata:
        return split_kawabata_data(inputs, test_cases, outputs, noise_stddevs)
    else:
        raise DataSetError(
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
    min_noise_stddev: float,
    outputs: StressOutputs,
) -> Tensor:
    noise_stddevs = relative_noise_stddevs * outputs
    return torch.where(
        noise_stddevs < min_noise_stddev,
        min_noise_stddev,
        noise_stddevs,
    )
