from typing import TypeAlias

import numpy as np
import pandas as pd

from bayesianmdisc.customtypes import Device, NPArray, PDDataFrame
from bayesianmdisc.data.base import (
    Data,
    assemble_test_case_identifiers,
    convert_to_torch,
    flatten_and_stack_arrays,
    numpy_data_type,
    stack_arrays,
)
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
    TestCaseIdentifier,
)
from bayesianmdisc.errors import DataError
from bayesianmdisc.io import ProjectDirectory

Component: TypeAlias = tuple[int, int]

irrelevant_stress_components = [4]


class LinkaHeartDataSet:
    def __init__(
        self,
        input_directory: str,
        project_directory: ProjectDirectory,
        device: Device,
        consider_shear_data: bool = True,
        consider_extension_data: bool = True,
    ):
        self._validate_data_configuration(consider_shear_data, consider_extension_data)
        self._consider_shear_data = consider_shear_data
        self._consider_biaxial_data = consider_extension_data
        self._input_directory = input_directory
        self._project_directory = project_directory
        self._device = device
        self._file_name = "CANNsHEARTdata_shear05.xlsx"
        self._excel_sheet_name = "Sheet1"
        self._row_offset = 3
        self._start_column_shear = 0
        self._start_column_biaxial = 15
        self._np_data_type = numpy_data_type
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ss_12 = test_case_identifier_simple_shear_12
        self._test_case_identifier_ss_21 = test_case_identifier_simple_shear_21
        self._test_case_identifier_ss_13 = test_case_identifier_simple_shear_13
        self._test_case_identifier_ss_31 = test_case_identifier_simple_shear_31
        self._test_case_identifier_ss_23 = test_case_identifier_simple_shear_23
        self._test_case_identifier_ss_32 = test_case_identifier_simple_shear_32
        self._data_frame = self._init_data_frame()

    def read_data(self) -> Data:
        all_deformation_gradients = []
        all_test_cases = []
        all_stresse_tensors = []

        if self._consider_shear_data:
            column = self._start_column_shear
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_12
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_13
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 3
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_21
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_23
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 3
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_31
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, self._test_case_identifier_ss_32
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

        if self._consider_biaxial_data:
            column = self._start_column_biaxial
            stretches_column, test_cases_colum, stresses_column = (
                self._read_biaxial_data(column)
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 5
            stretches_column, test_cases_colum, stresses_column = (
                self._read_biaxial_data(column)
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 5
            stretches_column, test_cases_colum, stresses_column = (
                self._read_biaxial_data(column)
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 5
            stretches_column, test_cases_colum, stresses_column = (
                self._read_biaxial_data(column)
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 5
            stretches_column, test_cases_colum, stresses_column = (
                self._read_biaxial_data(column)
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

        deformation_gradients = stack_arrays(all_deformation_gradients)
        test_cases = stack_arrays(all_test_cases)
        test_cases = test_cases.reshape((-1,))
        stress_tensors = stack_arrays(all_stresse_tensors)

        deformation_gradients_torch = convert_to_torch(
            deformation_gradients, self._device
        )
        test_cases_torch = convert_to_torch(test_cases, self._device)
        stresse_tensors_torch = convert_to_torch(stress_tensors, self._device)

        return deformation_gradients_torch, test_cases_torch, stresse_tensors_torch

    def _validate_data_configuration(
        self, consider_shear_data: bool, consider_biaxial_data: bool
    ) -> None:
        consider_any_data = consider_shear_data or consider_biaxial_data
        if not consider_any_data:
            raise DataError(
                """The data set is empty. 
                            Neither the shear nor the extension data are considered."""
            )

    def _init_data_frame(self) -> PDDataFrame:
        input_path = self._project_directory.get_input_file_path(
            file_name=self._file_name, subdir_name=self._input_directory
        )
        return pd.read_excel(input_path, sheet_name=self._excel_sheet_name)

    def _read_shear_data(
        self, start_column: int, shear_test_case_identifier: TestCaseIdentifier
    ) -> tuple[NPArray, NPArray, NPArray]:
        shear_strains = self._read_column(start_column)
        shear_stresses = self._read_column(start_column + 1)

        flattened_deformation_gradients = assemble_flattened_deformation_gradients(
            shear_strains, shear_test_case_identifier
        )
        test_cases = assemble_test_case_identifiers(
            shear_test_case_identifier, flattened_deformation_gradients
        )
        reduced_flattened_stress_tensors = assemble_reduced_flattened_stress_tensor(
            shear_stresses, shear_test_case_identifier
        )
        return (
            flattened_deformation_gradients,
            test_cases,
            reduced_flattened_stress_tensors,
        )

    def _read_biaxial_data(self, start_column: int) -> tuple[NPArray, NPArray, NPArray]:
        stretches_f = self._read_column(start_column).reshape((-1, 1))
        stresses_ff = self._read_column(start_column + 1).reshape((-1, 1))
        stretches_n = self._read_column(start_column + 2).reshape((-1, 1))
        stresses_nn = self._read_column(start_column + 3).reshape((-1, 1))
        stretches = np.hstack((stretches_f, stretches_n))
        stresses = np.hstack((stresses_ff, stresses_nn))

        flattened_deformation_gradients = assemble_flattened_deformation_gradients(
            stretches, self._test_case_identifier_bt
        )
        test_cases = assemble_test_case_identifiers(
            self._test_case_identifier_bt, flattened_deformation_gradients
        )
        reduced_flattened_stress_tensors = assemble_reduced_flattened_stress_tensor(
            stresses, self._test_case_identifier_bt
        )
        return (
            flattened_deformation_gradients,
            test_cases,
            reduced_flattened_stress_tensors,
        )

    def _read_column(self, column: int) -> NPArray:
        return (
            self._data_frame.iloc[self._row_offset :, column]
            .dropna()
            .astype(self._np_data_type)
            .values
        )


def assemble_flattened_deformation_gradients(
    deformation_inputs: NPArray, test_case_identifier: TestCaseIdentifier
) -> NPArray:

    def _assemble_one_biaxial_tension_deformation_gradient(
        deformation_input: NPArray,
    ) -> NPArray:
        stretch_f = deformation_input[0]
        stretch_n = deformation_input[1]
        stretch_s = 1.0 / (stretch_f * stretch_n)
        deformation_gradient = np.zeros((3, 3), dtype=numpy_data_type)
        deformation_gradient[0, 0] = stretch_f
        deformation_gradient[1, 1] = stretch_s
        deformation_gradient[2, 2] = stretch_n
        return deformation_gradient

    def _assemble_one_simple_shear_deformation_gradient(
        deformation_input: NPArray, test_case_identifier: TestCaseIdentifier
    ) -> NPArray:
        shear_strain = deformation_input
        shear_component = _map_to_shear_components(test_case_identifier)
        stretches = 1.0
        deformation_gradient = np.zeros((3, 3), dtype=numpy_data_type)
        deformation_gradient[0, 0] = stretches
        deformation_gradient[1, 1] = stretches
        deformation_gradient[2, 2] = stretches
        deformation_gradient[shear_component] = shear_strain
        return deformation_gradient

    deformation_gradients: list[NPArray] = []
    for deformation_input in deformation_inputs:
        if test_case_identifier == test_case_identifier_biaxial_tension:
            deformation_gradients += [
                _assemble_one_biaxial_tension_deformation_gradient(deformation_input)
            ]
        else:
            deformation_gradients += [
                _assemble_one_simple_shear_deformation_gradient(
                    deformation_input, test_case_identifier
                )
            ]

    return flatten_and_stack_arrays(deformation_gradients)


def assemble_reduced_flattened_stress_tensor(
    stress_outputs: NPArray, test_case_identifier: TestCaseIdentifier
) -> NPArray:

    def _assemble_one_biaxial_tension_stress_tensor(
        stress_output: NPArray,
    ) -> NPArray:
        stress_ff = stress_output[0]
        stress_nn = stress_output[1]
        stress_tensor = np.zeros((3, 3), dtype=numpy_data_type)
        stress_tensor[0, 0] = stress_ff
        stress_tensor[2, 2] = stress_nn
        return stress_tensor

    def _assemble_one_simple_shear_stress_tensor(
        stress_output: NPArray, test_case_identifier: TestCaseIdentifier
    ) -> NPArray:
        shear_stress = stress_output
        shear_component = _map_to_shear_components(test_case_identifier)
        symmetric_shear_component = tuple(reversed(shear_component))
        stress_tensor = np.zeros((3, 3), dtype=numpy_data_type)
        stress_tensor[shear_component] = shear_stress
        stress_tensor[symmetric_shear_component] = shear_stress
        return stress_tensor

    def _reduce_to_relevant_stresses(flattened_stress_tensors: NPArray) -> NPArray:
        return np.delete(flattened_stress_tensors, irrelevant_stress_components, 1)

    stress_tensors: list[NPArray] = []
    for stress_output in stress_outputs:
        if test_case_identifier == test_case_identifier_biaxial_tension:
            stress_tensors += [
                _assemble_one_biaxial_tension_stress_tensor(stress_output)
            ]
        else:
            stress_tensors += [
                _assemble_one_simple_shear_stress_tensor(
                    stress_output, test_case_identifier
                )
            ]

    flattened_stress_tensors = flatten_and_stack_arrays(stress_tensors)
    return _reduce_to_relevant_stresses(flattened_stress_tensors)


def _map_to_shear_components(
    shear_test_case_identifier: TestCaseIdentifier,
) -> Component:
    if shear_test_case_identifier == test_case_identifier_simple_shear_12:
        return (0, 1)
    elif shear_test_case_identifier == test_case_identifier_simple_shear_21:
        return (1, 0)
    elif shear_test_case_identifier == test_case_identifier_simple_shear_13:
        return (0, 2)
    elif shear_test_case_identifier == test_case_identifier_simple_shear_31:
        return (2, 0)
    elif shear_test_case_identifier == test_case_identifier_simple_shear_23:
        return (1, 2)
    elif shear_test_case_identifier == test_case_identifier_simple_shear_32:
        return (2, 1)
    else:
        raise DataError(f"Unvalid test case identifier: {shear_test_case_identifier}")
