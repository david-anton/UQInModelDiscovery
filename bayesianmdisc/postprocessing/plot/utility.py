import numpy as np

from bayesianmdisc.customtypes import NPArray
from bayesianmdisc.errors import PlotterError
from bayesianmdisc.testcases import (
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)


def split_treloar_inputs_and_outputs(
    inputs: NPArray, test_cases: NPArray, outputs: NPArray
) -> tuple[list[NPArray], list[int], list[NPArray]]:
    considered_test_cases = [
        test_case_identifier_uniaxial_tension,
        test_case_identifier_equibiaxial_tension,
        test_case_identifier_pure_shear,
    ]
    num_data_points_ut = 25
    num_data_points_ebt = 14
    num_data_points_ps = 14
    expected_set_sizes = [num_data_points_ut, num_data_points_ebt, num_data_points_ps]

    def split_data() -> tuple[list[NPArray], list[int], list[NPArray]]:
        input_sets = []
        test_case_identifiers = []
        output_sets = []

        for test_case in considered_test_cases:
            filter = test_cases == test_case
            input_sets += [inputs[filter]]
            test_case_identifiers += [test_case]
            output_sets += [outputs[filter]]

        return input_sets, test_case_identifiers, output_sets

    def validate_data_sets(
        input_sets: list[NPArray],
        test_case_identifiers: list[int],
        output_sets: list[NPArray],
    ) -> None:
        input_set_sizes = [len(set) for set in input_sets]
        output_set_sizes = [len(set) for set in output_sets]

        valid_set_sizes = (
            input_set_sizes == expected_set_sizes
            and len(test_case_identifiers) == 3
            and output_set_sizes == expected_set_sizes
        )
        valid_test_case_sets = test_case_identifiers == considered_test_cases

        if not valid_set_sizes and valid_test_case_sets:
            raise PlotterError(
                """The number of data points to be plotted
                                        does not match the size of the Treloar data set."""
            )

    input_sets, test_case_identifiers, output_sets = split_data()
    validate_data_sets(input_sets, test_case_identifiers, output_sets)
    return input_sets, test_case_identifiers, output_sets


def split_kawabata_inputs_and_outputs(
    inputs: NPArray, test_cases: NPArray, outputs: NPArray
) -> tuple[list[NPArray], list[int], list[NPArray]]:

    num_data_points = 76
    set_sizes = [1, 6, 6, 8, 8, 8, 8, 8, 7, 7, 6, 3]
    num_sets = len(set_sizes)

    def validate_data() -> None:
        num_inputs = len(inputs)
        num_test_cases = len(test_cases)
        num_outputs = len(outputs)
        valid_data = (
            num_inputs == num_data_points
            and num_test_cases == num_data_points
            and num_outputs == num_data_points
        )

        if not valid_data:
            raise PlotterError(
                f"""The input and/or output do not comprise the expected number of data points 
                    (input comprises {num_inputs} points, test cases {num_test_cases} points and 
                    output {num_outputs} but {num_data_points} data points are expected)."""
            )

    def determine_split_indices() -> list[int]:
        split_indices = [set_sizes[0]]
        for i in range(1, num_sets - 1):
            split_indices += [split_indices[-1] + set_sizes[i]]
        return split_indices

    def split_data_sets(
        split_indices: list[int],
    ) -> tuple[list[NPArray], list[int], list[NPArray]]:
        input_sets = np.split(inputs, split_indices)
        test_case_sets = np.split(test_cases, split_indices)
        test_case_identifiers = [
            int(test_case_set[0]) for test_case_set in test_case_sets
        ]
        output_sets = np.split(outputs, split_indices)
        return input_sets, test_case_identifiers, output_sets

    validate_data()
    split_indices = determine_split_indices()
    return split_data_sets(split_indices)


def split_linka_inputs_and_outputs(
    inputs: NPArray, test_cases: NPArray, outputs: NPArray
) -> tuple[
    list[NPArray],
    list[int],
    list[NPArray],
]:
    num_data_sets = 11
    num_data_points = 121

    def validate_data() -> None:
        num_inputs = len(inputs)
        num_test_cases = len(test_cases)
        num_outputs = len(outputs)
        valid_data = (
            num_inputs == num_data_points
            and num_test_cases == num_data_points
            and num_outputs == num_data_points
        )

        if not valid_data:
            raise PlotterError(
                f"""The input and/or output do not comprise the expected number of data points 
                    (input comprises {num_inputs} points, test cases {num_test_cases} points and 
                    output {num_outputs} but {num_data_points} data points are expected)."""
            )

    def split_data_sets() -> tuple[list[NPArray], list[int], list[NPArray]]:
        input_sets = np.split(inputs, num_data_sets, axis=0)
        test_case_sets = np.split(test_cases, num_data_sets, axis=0)
        test_case_identifiers = [
            int(test_case_set[0]) for test_case_set in test_case_sets
        ]
        output_sets = np.split(outputs, num_data_sets, axis=0)
        return input_sets, test_case_identifiers, output_sets

    validate_data()
    return split_data_sets()
