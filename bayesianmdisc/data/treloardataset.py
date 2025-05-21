from typing import TypeAlias

import numpy as np

from bayesianmdisc.customtypes import Device, NPArray
from bayesianmdisc.data.base import (
    Data,
    DeformationInputs,
    numpy_data_type,
    stack_arrays,
    convert_to_torch,
    assemble_test_case_identifiers,
)
from bayesianmdisc.data.testcases import (
    TestCases,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader


class TreloarDataSet:
    def __init__(
        self,
        input_directory: str,
        project_directory: ProjectDirectory,
        device: Device,
    ):
        self._input_directory = input_directory
        self._project_directory = project_directory
        self._device = device
        self._csv_reader = CSVDataReader(self._project_directory)
        self._file_name_uniaxial_tension = "TreloarDataUT.csv"
        self._file_name_equibiaxial_tension = "TreloarDataEBT.csv"
        self._file_name_pure_shear = "TreloarDataPS.csv"
        self._index_stretch = 0
        self._index_stresses = 1
        self._np_data_type = numpy_data_type
        self._test_case_identifier_ut = test_case_identifier_uniaxial_tension
        self._test_case_identifier_ebt = test_case_identifier_equibiaxial_tension
        self._test_case_identifier_ps = test_case_identifier_pure_shear

    def read_data(self) -> Data:
        stretches_ut, test_cases_ut, stresses_ut = self._read_data(
            self._file_name_uniaxial_tension, self._test_case_identifier_ut
        )
        stretches_ebt, test_cases_ebt, stresses_ebt = self._read_data(
            self._file_name_equibiaxial_tension, self._test_case_identifier_ebt
        )
        stretches_ps, test_cases_ps, stresses_ps = self._read_data(
            self._file_name_pure_shear, self._test_case_identifier_ps
        )

        stretches = stack_arrays([stretches_ut, stretches_ebt, stretches_ps])
        test_cases = stack_arrays([test_cases_ut, test_cases_ebt, test_cases_ps])
        test_cases = test_cases.reshape((-1,))
        stresses = stack_arrays([stresses_ut, stresses_ebt, stresses_ps])

        stretches_torch = convert_to_torch(stretches, self._device)
        test_cases_torch = convert_to_torch(test_cases, self._device)
        stresses_torch = convert_to_torch(stresses, self._device)

        return stretches_torch, test_cases_torch, stresses_torch

    def generate_uniform_inputs(
        self, num_points_per_test_case: int
    ) -> tuple[DeformationInputs, TestCases]:
        stretches_ut, test_cases_ut = self._generate_uniform_inputs(
            self._file_name_uniaxial_tension,
            self._test_case_identifier_ut,
            num_points_per_test_case,
        )
        stretches_ebt, test_cases_ebt = self._generate_uniform_inputs(
            self._file_name_equibiaxial_tension,
            self._test_case_identifier_ebt,
            num_points_per_test_case,
        )
        stretches_ps, test_cases_ps = self._generate_uniform_inputs(
            self._file_name_pure_shear,
            self._test_case_identifier_ps,
            num_points_per_test_case,
        )
        test_cases = stack_arrays([test_cases_ut, test_cases_ebt, test_cases_ps])
        test_cases = test_cases.reshape((-1,))
        stretches = stack_arrays([stretches_ut, stretches_ebt, stretches_ps])

        stretches_torch = convert_to_torch(stretches, self._device)
        test_cases_torch = convert_to_torch(test_cases, self._device)

        return stretches_torch, test_cases_torch

    def _read_data(
        self, file_name: str, test_case_identifier: int
    ) -> tuple[NPArray, NPArray, NPArray]:
        data = self._read_csv_file(file_name)
        stretch_factors = data[:, self._index_stretch].reshape((-1, 1))
        stretches = self._calculate_stretches(stretch_factors, test_case_identifier)
        test_cases = assemble_test_case_identifiers(test_case_identifier, stretches)
        stresses = data[:, self._index_stresses].reshape((-1, 1))
        return stretches, test_cases, stresses

    def _generate_uniform_inputs(
        self, file_name: str, test_case_identifier: int, num_inputs: int
    ) -> tuple[NPArray, NPArray]:
        data = self._read_csv_file(file_name)
        data_stretch_factors = data[:, self._index_stretch]
        min_stretch_factor = np.amin(data_stretch_factors)
        max_stretch_factor = np.amax(data_stretch_factors)
        stretch_factors = np.linspace(
            min_stretch_factor, max_stretch_factor, num=num_inputs, endpoint=True
        ).reshape((-1, 1))
        stretches = self._calculate_stretches(stretch_factors, test_case_identifier)
        test_cases = assemble_test_case_identifiers(test_case_identifier, stretches)
        return stretches, test_cases

    def _read_csv_file(self, file_name: str) -> NPArray:
        return self._csv_reader.read(
            file_name=file_name, subdir_name=self._input_directory, seperator=";"
        )

    def _calculate_stretches(
        self, stretch_factors: NPArray, test_case_identifier: int
    ) -> NPArray:
        one = np.array(1.0)
        if test_case_identifier == self._test_case_identifier_ut:
            stretches_1 = stretch_factors
            stretches_2 = stretches_3 = one / np.sqrt(stretch_factors)
        elif test_case_identifier == self._test_case_identifier_ebt:
            stretches_1 = stretches_2 = stretch_factors
            stretches_3 = one / stretch_factors**2
        else:
            stretches_1 = stretch_factors
            stretches_2 = np.ones_like(stretch_factors)
            stretches_3 = one / stretch_factors

        return np.hstack((stretches_1, stretches_2, stretches_3))
