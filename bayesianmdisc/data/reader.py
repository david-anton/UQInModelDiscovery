import os
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
import pandas as pd
from torch import vmap

from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.types import NPArray, Tensor, PDDataFrame

UrreatoData: TypeAlias = tuple[Tensor, Tensor, Tensor]


# class UrreaSyntheticDataReader:
#     def __init__(self, input_directory: str, project_directory: ProjectDirectory):
#         self._input_directory = input_directory
#         self._project_directory = project_directory

#     def read(self) -> UrreatoData:
#         print("Start reading synthetic data from Linka et al.")
#         input_directory_path = self._join_input_directory_path()
#         deformation_gradients_list = []
#         flattened_stresses_list = []
#         stretches_list = []

#         for file_name in sorted(os.listdir(input_directory_path)):
#             if self._validate_file_name(file_name):
#                 data = self._read_data_from_file(file_name)

#                 if data is None or (len(data) < 11):
#                     continue

#                 # data => [offset, F11, F12, F21, F22, F33, P11, P22, P33, P12, P21]
#                 F11, F12, F21, F22, F33 = data[1:6]
#                 P11, P22, P33, P12, P21 = data[6:11]

#                 deformation_gradient = np.eye(3)
#                 deformation_gradient[0, 0] = F11
#                 deformation_gradient[0, 1] = F12
#                 deformation_gradient[1, 0] = F21
#                 deformation_gradient[1, 1] = F22
#                 deformation_gradient[2, 2] = F33
#                 deformation_gradients_list.append(deformation_gradient)

#                 stretches = np.array([F11, F22])
#                 stretches_list.append(stretches)

#                 flattened_stress_tensor = np.array([P11, P22, P33, P12, P21])
#                 flattened_stresses_list.append(flattened_stress_tensor)

#         deformation_gradients_data = torch.from_numpy(
#             np.stack(deformation_gradients_list),
#         ).type(torch.get_default_dtype())
#         stretches_data = torch.from_numpy(
#             np.stack(stretches_list),
#         ).type(torch.get_default_dtype())
#         flattened_stresses_data = torch.from_numpy(
#             np.stack(flattened_stresses_list),
#         ).type(torch.get_default_dtype())

#         num_samples = len(deformation_gradients_list)
#         self._validate_number_of_samples(num_samples)
#         return (
#             deformation_gradients_data,
#             stretches_data,
#             flattened_stresses_data,
#         )

#     def _join_input_directory_path(self) -> Path:
#         return self._project_directory.get_input_subdirectory_path(
#             subdir_name=self._input_directory
#         )

#     def _validate_file_name(self, file_name: str) -> bool:
#         return (
#             file_name.endswith(".m")
#             and "Lambda1_" in file_name
#             and "Lambda2_" in file_name
#         )

#     def _read_data_from_file(self, file_name: str) -> NPArray | None:
#         input_path = self._join_input_file_path(file_name)
#         try:
#             with open(input_path, "r") as file:
#                 lines = file.readlines()
#                 data = np.array([float(line.strip()) for line in lines if line.strip()])
#             return data
#         except Exception as exception:
#             print(f"Error reading {input_path}: {exception}")
#             return None

#     def _join_input_file_path(self, file_name: str) -> Path:
#         return self._project_directory.get_input_file_path(
#             file_name=file_name, subdir_name=self._input_directory
#         )

#     def _validate_number_of_samples(self, num_samples: int) -> None:
#         print(f"Found {num_samples} samples in {self._input_directory}")
#         if num_samples == 0:
#             raise RuntimeError("No valid data loaded!")


DeformationGradients: TypeAlias = Tensor
StressTensors: TypeAlias = Tensor
HydrostaticPressure: TypeAlias = Tensor
LinkaHeartData: TypeAlias = tuple[
    DeformationGradients, StressTensors, HydrostaticPressure
]


class LinkaHeartDataReader:
    def __init__(
        self, file_name: str, input_directory: str, project_directory: ProjectDirectory
    ):
        self._file_name = file_name
        self._input_directory = input_directory
        self._project_directory = project_directory
        self._excel_sheet_name = "Sheet1"
        self._row_offset = 3
        self._start_column_shear = 0
        self._start_column_biaxial = 15
        self._np_data_type = np.float64
        self._data_frame = self._init_data_frame()

    def read(self) -> LinkaHeartData:
        (
            deformation_gradients_shear,
            cauchy_stress_tensors_shear,
            hydrostatic_pressures_shear,
        ) = self.read_shear_data()
        (
            deformation_gradients_biaxial,
            cauchy_stress_tensors_biaxial,
            hydrostatic_pressures_biaxial,
        ) = self.read_biaxial_data()

        deformation_gradients = torch.concat(
            (deformation_gradients_shear, deformation_gradients_biaxial), dim=0
        )
        cauchy_stress_tensors = torch.concat(
            (cauchy_stress_tensors_shear, cauchy_stress_tensors_biaxial), dim=0
        )
        hydrostatic_pressures = torch.concat(
            (hydrostatic_pressures_shear, hydrostatic_pressures_biaxial), dim=0
        )
        return deformation_gradients, cauchy_stress_tensors, hydrostatic_pressures

    def read_shear_data(self) -> LinkaHeartData:
        all_deformation_gradients = []
        all_cauchy_stress_tensors = []

        def read_data(
            start_column: int, tensor_row: int, tensor_column: int
        ) -> tuple[list[NPArray], list[NPArray]]:
            deformation_gradients: list[NPArray] = []
            cauchy_stress_tensors: list[NPArray] = []
            gammas = self._read_column(start_column)
            sigmas = self._read_column(start_column + 1)

            for gamma, sigma in zip(gammas, sigmas):
                deformation_gradient = np.eye(3, dtype=self._np_data_type)
                deformation_gradient[tensor_row, tensor_column] = gamma
                deformation_gradients += [deformation_gradient]
                cauchy_stress_tensor = np.zeros((3, 3), dtype=self._np_data_type)
                cauchy_stress_tensor[tensor_row, tensor_column] = sigma
                cauchy_stress_tensor[tensor_column, tensor_row] = sigma
                cauchy_stress_tensors += [cauchy_stress_tensor]
            return deformation_gradients, cauchy_stress_tensors

        column = self._start_column_shear
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=0, tensor_column=1
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 2
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=0, tensor_column=2
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 3
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=1, tensor_column=0
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 2
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=1, tensor_column=2
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 3
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=2, tensor_column=0
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 2
        deformation_gradients, cauchy_stress_tensors = read_data(
            column, tensor_row=2, tensor_column=1
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        all_hydrostatic_pressures = self._calculate_hydrostatic_pressure(
            all_cauchy_stress_tensors
        )

        deformation_gradients_torch = torch.stack(
            [
                self._convert_to_torch_tensor(array)
                for array in all_deformation_gradients
            ],
            dim=0,
        )
        cauchy_stress_tensors_torch = torch.stack(
            [
                self._convert_to_torch_tensor(array)
                for array in all_cauchy_stress_tensors
            ],
            dim=0,
        )
        hydrostatic_pressures_torch = self._convert_to_torch_tensor(
            all_hydrostatic_pressures
        )

        return (
            deformation_gradients_torch,
            cauchy_stress_tensors_torch,
            hydrostatic_pressures_torch,
        )

    def read_biaxial_data(self) -> LinkaHeartData:
        all_deformation_gradients = []
        all_cauchy_stress_tensors = []

        def add_data(
            start_column: int, ratio_fiber: float, ratio_normal: float
        ) -> tuple[list[NPArray], list[NPArray]]:
            deformation_gradients: list[NPArray] = []
            cauchy_stress_tensors: list[NPArray] = []
            stretches = self._read_column(start_column)
            sigmas_fiber = self._read_column(start_column + 1)
            sigmas_normal = self._read_column(start_column + 3)

            for stretch, sigma_fiber, sigma_normal in zip(
                stretches, sigmas_fiber, sigmas_normal
            ):
                stretch_fiber = ratio_fiber * stretch
                stretch_normal = ratio_normal * stretch
                stretch_sheet = 1 / (stretch_fiber * stretch_normal)

                deformation_gradient = np.zeros((3, 3), dtype=self._np_data_type)
                deformation_gradient[0, 0] = stretch_fiber
                deformation_gradient[1, 1] = stretch_sheet
                deformation_gradient[2, 2] = stretch_normal
                deformation_gradients += [deformation_gradient]
                cauchy_stress_tensor = np.zeros((3, 3), dtype=self._np_data_type)
                cauchy_stress_tensor[0, 0] = sigma_fiber
                cauchy_stress_tensor[2, 2] = sigma_normal
                cauchy_stress_tensors += [cauchy_stress_tensor]
            return deformation_gradients, cauchy_stress_tensors

        column = self._start_column_biaxial
        deformation_gradients, cauchy_stress_tensors = add_data(
            column, ratio_fiber=1.0, ratio_normal=1.0
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 5
        deformation_gradients, cauchy_stress_tensors = add_data(
            column, ratio_fiber=1.0, ratio_normal=0.75
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 5
        deformation_gradients, cauchy_stress_tensors = add_data(
            column, ratio_fiber=0.75, ratio_normal=1.0
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 5
        deformation_gradients, cauchy_stress_tensors = add_data(
            column, ratio_fiber=1.0, ratio_normal=0.5
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        column = column + 5
        deformation_gradients, cauchy_stress_tensors = add_data(
            column, ratio_fiber=0.5, ratio_normal=1.0
        )
        all_deformation_gradients += deformation_gradients
        all_cauchy_stress_tensors += cauchy_stress_tensors

        all_hydrostatic_pressures = self._calculate_hydrostatic_pressure(
            all_cauchy_stress_tensors
        )

        deformation_gradients_torch = torch.stack(
            [
                self._convert_to_torch_tensor(array)
                for array in all_deformation_gradients
            ],
            dim=0,
        )
        cauchy_stress_tensors_torch = torch.stack(
            [
                self._convert_to_torch_tensor(array)
                for array in all_cauchy_stress_tensors
            ],
            dim=0,
        )
        hydrostatic_pressures_torch = self._convert_to_torch_tensor(
            all_hydrostatic_pressures
        )

        return (
            deformation_gradients_torch,
            cauchy_stress_tensors_torch,
            hydrostatic_pressures_torch,
        )

    def _calculate_hydrostatic_pressure(
        self, cauchy_stress_tensors: list[NPArray]
    ) -> NPArray:
        return np.array(
            [-1 / 3 * np.trace(array) for array in cauchy_stress_tensors],
            dtype=self._np_data_type,
        )

    def _init_data_frame(self) -> PDDataFrame:
        input_path = self._join_input_path()
        return pd.read_excel(input_path, sheet_name=self._excel_sheet_name)

    def _join_input_path(self) -> Path:
        return self._project_directory.get_input_file_path(
            file_name=self._file_name, subdir_name=self._input_directory
        )

    def _read_column(self, column: int) -> Tensor:
        return (
            self._data_frame.iloc[self._row_offset :, column]
            .dropna()
            .astype(self._np_data_type)
            .values
        )

    def _convert_to_torch_tensor(self, array: NPArray) -> Tensor:
        return torch.from_numpy(array).type(torch.get_default_dtype())
