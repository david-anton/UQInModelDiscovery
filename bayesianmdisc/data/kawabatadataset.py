import numpy as np

from bayesianmdisc.customtypes import Device, NPArray
from bayesianmdisc.data.base import (
    Data,
    assemble_test_case_identifiers,
    convert_to_torch,
    numpy_data_type,
)
from bayesianmdisc.data.testcases import test_case_identifier_biaxial_tension
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader


class KawabataDataSet:
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

    def read_data(self) -> Data:
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
