from typing import TypeAlias

import numpy as np
import pandas as pd

from bayesianmdisc.customtypes import Device, NPArray, PDDataFrame
from bayesianmdisc.data.base import (
    Data,
    stack_arrays,
    convert_to_torch,
    assemble_test_case_identifiers,
    flatten_and_stack_arrays,
    NPArrayList,
    numpy_data_type,
)
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_32,
)
from bayesianmdisc.errors import DataError
from bayesianmdisc.io import ProjectDirectory

Component: TypeAlias = tuple[int, int]


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
        self._irrelevant_stress_components = [4]
        self._data_frame = self._init_data_frame()

    def read_data(self) -> Data:
        all_deformation_gradients = []
        all_test_cases = []
        all_stresse_tensors = []

        if self._consider_shear_data:
            column = self._start_column_shear
            component = (0, 1)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            component = (0, 2)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 3
            component = (1, 0)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            component = (1, 2)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 3
            component = (2, 0)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
            )
            all_deformation_gradients.append(stretches_column)
            all_test_cases.append(test_cases_colum)
            all_stresse_tensors.append(stresses_column)

            column = column + 2
            component = (2, 1)
            stretches_column, test_cases_colum, stresses_column = self._read_shear_data(
                column, component
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
        self, start_column: int, stress_component: Component
    ) -> tuple[NPArray, NPArray, NPArray]:
        symmetric_stress_component = tuple(reversed(stress_component))
        deformation_gradients: NPArrayList = []
        stress_tensors: NPArrayList = []
        shear_strains = self._read_column(start_column)
        shear_stresses = self._read_column(start_column + 1)

        for shear_strain, shear_stress in zip(shear_strains, shear_stresses):
            stretches = 1.0
            deformation_gradient = np.zeros((3, 3), dtype=self._np_data_type)
            deformation_gradient[0, 0] = stretches
            deformation_gradient[1, 1] = stretches
            deformation_gradient[2, 2] = stretches
            deformation_gradient[symmetric_stress_component] = shear_strain
            deformation_gradients += [deformation_gradient]
            stress_tensor = np.zeros((3, 3), dtype=self._np_data_type)
            stress_tensor[stress_component] = shear_stress
            stress_tensor[symmetric_stress_component] = shear_stress
            stress_tensors += [stress_tensor]

        flattened_deformation_gradients = flatten_and_stack_arrays(
            deformation_gradients
        )
        test_case_identifier = self._map_to_shear_test_case_identifier(stress_component)
        test_cases = assemble_test_case_identifiers(
            test_case_identifier, flattened_deformation_gradients
        )
        flattened_stress_tensors = flatten_and_stack_arrays(stress_tensors)
        reduced_flattened_stress_tensors = self._reduce_to_relevant_stresses(
            flattened_stress_tensors
        )
        return (
            flattened_deformation_gradients,
            test_cases,
            reduced_flattened_stress_tensors,
        )

    def _map_to_shear_test_case_identifier(self, stress_component: Component) -> int:
        if stress_component == (0, 1):
            return self._test_case_identifier_ss_12
        elif stress_component == (1, 0):
            return self._test_case_identifier_ss_21
        elif stress_component == (0, 2):
            return self._test_case_identifier_ss_13
        elif stress_component == (2, 0):
            return self._test_case_identifier_ss_31
        elif stress_component == (1, 2):
            return self._test_case_identifier_ss_23
        elif stress_component == (2, 1):
            return self._test_case_identifier_ss_32
        else:
            raise DataError(f"Unvalid stress component: {stress_component}")

    def _reduce_to_relevant_stresses(self, flattened_stress_tensor: NPArray) -> NPArray:
        return np.delete(flattened_stress_tensor, self._irrelevant_stress_components, 1)

    def _read_biaxial_data(self, start_column: int) -> tuple[NPArray, NPArray, NPArray]:
        deformation_gradients: NPArrayList = []
        stress_tensors: NPArrayList = []
        stretches_f = self._read_column(start_column)
        stresses_ff = self._read_column(start_column + 1)
        stretches_n = self._read_column(start_column + 2)
        stresses_nn = self._read_column(start_column + 3)

        for stretch_f, stress_ff, stretch_n, stress_nn in zip(
            stretches_f, stresses_ff, stretches_n, stresses_nn
        ):
            stretch_s = 1.0 / (stretch_f * stretch_n)
            deformation_gradient = np.zeros((3, 3), dtype=self._np_data_type)
            deformation_gradient[0, 0] = stretch_f
            deformation_gradient[1, 1] = stretch_s
            deformation_gradient[2, 2] = stretch_n
            deformation_gradients += [deformation_gradient]
            stress_tensor = np.zeros((3, 3), dtype=self._np_data_type)
            stress_tensor[0, 0] = stress_ff
            stress_tensor[2, 2] = stress_nn
            stress_tensors += [stress_tensor]

        flattened_deformation_gradients = flatten_and_stack_arrays(
            deformation_gradients
        )
        test_cases = assemble_test_case_identifiers(
            self._test_case_identifier_bt, flattened_deformation_gradients
        )
        flattened_stress_tensors = flatten_and_stack_arrays(stress_tensors)
        return flattened_deformation_gradients, test_cases, flattened_stress_tensors

    def _read_column(self, column: int) -> NPArray:
        return (
            self._data_frame.iloc[self._row_offset :, column]
            .dropna()
            .astype(self._np_data_type)
            .values
        )
