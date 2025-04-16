from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import pandas as pd
import torch

from bayesianmdisc.customtypes import Device, NPArray, PDDataFrame, Tensor
from bayesianmdisc.data.base import (
    DeformationInputs,
    NPArrayList,
    StressOutputs,
    numpy_data_type,
)
from bayesianmdisc.data.testcases import (
    TestCases,
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader

Data: TypeAlias = tuple[DeformationInputs, TestCases, StressOutputs]


class DataReaderProtocol(Protocol):
    def read(self) -> Data: ...


class TreloarDataReader:
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

    def read(self) -> Data:
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

    def _read_data(
        self, file_name: str, test_case_identifier: int
    ) -> tuple[NPArray, NPArray, NPArray]:
        data = self._read_csv_file(file_name)
        stretch_factors = data[:, self._index_stretch].reshape((-1, 1))
        stretches = self._calculate_stretches(stretch_factors, test_case_identifier)
        test_cases = assemble_test_case_identifiers(test_case_identifier, stretches)
        stresses = data[:, self._index_stresses].reshape((-1, 1))
        return stretches, test_cases, stresses

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


class KawabataDataReader:
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
        self._file_name = "Kawabata.csv"
        self._slice_stretches = slice(0, 2)
        self._slice_stresses = slice(2, 4)
        self._np_data_type = numpy_data_type
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension

    def read(self) -> Data:
        stretches, test_cases, stresses = self._read_data()
        stretches_torch = convert_to_torch(stretches, self._device)
        test_cases_torch = convert_to_torch(test_cases, self._device)
        stresses_torch = convert_to_torch(stresses, self._device)
        return stretches_torch, test_cases_torch, stresses_torch

    def _read_data(self) -> tuple[NPArray, NPArray, NPArray]:
        data = self._read_csv_file()
        biaxial_stretches = data[:, self._slice_stretches]
        stretches = self._calculate_stretches(biaxial_stretches)
        test_cases = assemble_test_case_identifiers(
            self._test_case_identifier_bt, stretches
        )
        stresses = data[:, self._slice_stresses]
        return stretches, test_cases, stresses

    def _read_csv_file(self) -> NPArray:
        return self._csv_reader.read(
            file_name=self._file_name, subdir_name=self._input_directory, seperator=";"
        )

    def _calculate_stretches(self, biaxial_stretches: NPArray) -> NPArray:
        one = np.array(1.0)
        stretches_1 = biaxial_stretches[:, 0].reshape((-1, 1))
        stretches_2 = biaxial_stretches[:, 1].reshape((-1, 1))
        stretches_3 = one / (stretches_1 * stretches_2)
        return np.hstack((stretches_1, stretches_2, stretches_3))


class LinkaHeartDataReader:
    def __init__(
        self,
        input_directory: str,
        project_directory: ProjectDirectory,
        device: Device,
    ):
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
        self._data_frame = self._init_data_frame()

    def read(self) -> Data:
        all_stretches = []
        all_test_cases = []
        all_stresses = []

        column = self._start_column_biaxial
        stretches_column, test_cases_colum, stresses_column = self._read_data(column)
        all_stretches.append(stretches_column)
        all_test_cases.append(test_cases_colum)
        all_stresses.append(stresses_column)

        column = column + 5
        stretches_column, test_cases_colum, stresses_column = self._read_data(column)
        all_stretches.append(stretches_column)
        all_test_cases.append(test_cases_colum)
        all_stresses.append(stresses_column)

        column = column + 5
        stretches_column, test_cases_colum, stresses_column = self._read_data(column)
        all_stretches.append(stretches_column)
        all_test_cases.append(test_cases_colum)
        all_stresses.append(stresses_column)

        column = column + 5
        stretches_column, test_cases_colum, stresses_column = self._read_data(column)
        all_stretches.append(stretches_column)
        all_test_cases.append(test_cases_colum)
        all_stresses.append(stresses_column)

        column = column + 5
        stretches_column, test_cases_colum, stresses_column = self._read_data(column)
        all_stretches.append(stretches_column)
        all_test_cases.append(test_cases_colum)
        all_stresses.append(stresses_column)

        stretches = stack_arrays(all_stretches)
        test_cases = stack_arrays(all_test_cases)
        test_cases = test_cases.reshape((-1,))
        stresses = stack_arrays(all_stresses)

        stretches_torch = convert_to_torch(stretches, self._device)
        test_cases_torch = convert_to_torch(test_cases, self._device)
        stresses_torch = convert_to_torch(stresses, self._device)

        return stretches_torch, test_cases_torch, stresses_torch

    def _init_data_frame(self) -> PDDataFrame:
        input_path = self._join_input_path()
        return pd.read_excel(input_path, sheet_name=self._excel_sheet_name)

    def _join_input_path(self) -> Path:
        return self._project_directory.get_input_file_path(
            file_name=self._file_name, subdir_name=self._input_directory
        )

    def _read_data(self, start_column: int) -> tuple[NPArray, NPArray, NPArray]:
        all_stretches: NPArrayList = []
        all_stresses: NPArrayList = []
        stretches_fiber = self._read_column(start_column)
        stresses_fiber = self._read_column(start_column + 1)
        stretches_normal = self._read_column(start_column + 2)
        stresses_normal = self._read_column(start_column + 3)

        for stretch_fiber, stress_fiber, stretch_normal, stress_normal in zip(
            stretches_fiber, stresses_fiber, stretches_normal, stresses_normal
        ):
            stretches = np.array(
                [stretch_fiber, stretch_normal],
                dtype=self._np_data_type,
            )
            all_stretches += [stretches]
            stresses = np.array([stress_fiber, stress_normal], dtype=self._np_data_type)
            all_stresses += [stresses]

        stretches = np.vstack(all_stretches)
        test_cases = assemble_test_case_identifiers(
            self._test_case_identifier_bt, stretches
        )
        stresses = np.vstack(all_stresses)
        return stretches, test_cases, stresses

    def _read_column(self, column: int) -> NPArray:
        return (
            self._data_frame.iloc[self._row_offset :, column]
            .dropna()
            .astype(self._np_data_type)
            .values
        )


def assemble_test_case_identifiers(
    test_case_identifier: int, deformation_input: NPArray
) -> NPArray:
    num_stretches = len(deformation_input)
    return np.full((num_stretches, 1), test_case_identifier, dtype=np.int64)


def convert_to_torch(array: NPArray, device: Device) -> Tensor:
    return torch.from_numpy(array).type(torch.get_default_dtype()).to(device)


def flatten_and_stack_arrays(arrays: list[NPArray]) -> NPArray:
    return stack_arrays([flatten_array(array) for array in arrays])


def flatten_array(array: NPArray) -> NPArray:
    return array.reshape(-1)


def stack_arrays(arrays: list[NPArray]) -> NPArray:
    return np.vstack(arrays)
