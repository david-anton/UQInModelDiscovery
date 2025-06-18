from typing import TypeAlias

from .testcases import (
    test_case_identifier_uniaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_biaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_32,
    TestCaseIdentifier,
)
from .customtypes import Tensor
from .errors import DataError

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
    [test_case_identifier_simple_shear_21, test_case_identifier_simple_shear_12],
    [test_case_identifier_simple_shear_31, test_case_identifier_simple_shear_13],
    [test_case_identifier_simple_shear_12, test_case_identifier_simple_shear_21],
    [test_case_identifier_simple_shear_32, test_case_identifier_simple_shear_23],
    [test_case_identifier_simple_shear_13, test_case_identifier_simple_shear_31],
    [test_case_identifier_simple_shear_23, test_case_identifier_simple_shear_32],
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
