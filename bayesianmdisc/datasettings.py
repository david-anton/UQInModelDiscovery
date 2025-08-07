from dataclasses import dataclass
from typing import TypeAlias

import torch

from .customtypes import Device, Tensor
from .errors import DataError
from .testcases import (
    TestCaseIdentifier,
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
    test_case_identifier_uniaxial_tension,
)

data_set_label_treloar = "treloar"
data_set_label_kawabata = "kawabata"
data_set_label_linka = "heart_data_linka"
data_set_label_synthetic_linka = "synthetic_heart_data_linka"


SkippedInputIndices: TypeAlias = list[int]
zero_stress_inputs_treloar = [0, 25, 39]
num_data_sets_treloar = 11


def determine_skipped_input_indices(
    data_set_label, inputs: Tensor
) -> SkippedInputIndices:
    if data_set_label == data_set_label_treloar:
        return zero_stress_inputs_treloar
    elif (
        data_set_label == data_set_label_linka
        or data_set_label == data_set_label_synthetic_linka
    ):
        num_inputs = len(inputs)
        num_points_per_data_set = int(round(num_inputs / num_data_sets_treloar))
        return [
            num_points_per_data_set * index_data_set
            for index_data_set in range(num_data_sets_treloar)
        ]
    else:
        raise DataError("""There is no implementation for the specified data set""")


RelevantTestCases: TypeAlias = list[list[TestCaseIdentifier]]
relevant_test_cases_for_sensitivity_analysis_treloar = [
    [
        test_case_identifier_uniaxial_tension,
        test_case_identifier_equibiaxial_tension,
        test_case_identifier_pure_shear,
    ]
]
relevant_test_cases_for_sensitivity_analysis_linka = [
    [test_case_identifier_biaxial_tension],
    [test_case_identifier_simple_shear_21],
    [test_case_identifier_simple_shear_31],
    [test_case_identifier_simple_shear_12],
    [test_case_identifier_simple_shear_32],
    [test_case_identifier_simple_shear_13],
    [test_case_identifier_simple_shear_23],
    [test_case_identifier_biaxial_tension],
]


def determine_relevant_test_cases_for_outputs(data_set_label) -> RelevantTestCases:
    if data_set_label == data_set_label_treloar:
        return relevant_test_cases_for_sensitivity_analysis_treloar
    elif (
        data_set_label == data_set_label_linka
        or data_set_label == data_set_label_synthetic_linka
    ):
        return relevant_test_cases_for_sensitivity_analysis_linka
    else:
        raise DataError(f"""There is no implementation for the specified data set""")


@dataclass
class LinkasModelParameters:
    names: tuple[str, ...]
    values: tuple[float, ...]


def create_four_terms_linka_model_parameters() -> LinkasModelParameters:
    parameter_names = (
        "W_2_7 (l2, I_2, p2, I)",
        "W_1_12 (l1, I_4f, p2, exp)",
        "W_2_12 (l2, I_4f, p2, exp)",
        "W_1_20 (l1, I_4n, p2, exp)",
        "W_2_20 (l2, I_4n, p2, exp)",
        "W_1_24 (l1, I_8fs, p2, exp)",
        "W_2_24 (l2, I_8fs, p2, exp)",
    )
    parameter_values = (5.162, 21.151, 0.081, 4.371, 0.315, 0.508, 0.486)
    return LinkasModelParameters(names=parameter_names, values=parameter_values)


def assemble_input_mask_for_treloar(device: Device) -> Tensor | None:
    return torch.tensor([True, True, False], device=device)


def assemble_input_masks_for_linka(
    device: Device,
) -> tuple[Tensor, ...] | tuple[None, ...]:

    def create_mask(true_indices: list[int]) -> Tensor:
        mask = torch.full((9,), False, device=device)
        for index in true_indices:
            mask[index] = True
        return mask

    input_mask_sigma_ff = create_mask([0, 8])
    input_mask_sigma_fs = create_mask([3])
    input_mask_sigma_fn = create_mask([6])

    input_mask_sigma_sf = create_mask([1])
    input_mask_sigma_sn = create_mask([7])

    input_mask_sigma_nf = create_mask([2])
    input_mask_sigma_ns = create_mask([5])
    input_mask_sigma_nn = create_mask([0, 8])

    return (
        input_mask_sigma_ff,
        input_mask_sigma_fs,
        input_mask_sigma_fn,
        input_mask_sigma_sf,
        input_mask_sigma_sn,
        input_mask_sigma_nf,
        input_mask_sigma_ns,
        input_mask_sigma_nn,
    )
