import os
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.types import NPArray, Tensor

LinkaData: TypeAlias = tuple[Tensor, Tensor, Tensor]


class LinkaDataReader:
    def __init__(self, input_directory: str, project_directory: ProjectDirectory):
        self._input_directory = input_directory
        self._project_directory = project_directory

    def read(self) -> LinkaData:
        input_directory_path = self._join_input_directory_path()
        deformation_gradients_list = []
        flattened_stresses_list = []
        stretches_list = []

        for file_name in sorted(os.listdir(input_directory_path)):
            if self._validate_file_name(file_name):
                data = self._read_data_from_file(file_name)

                if data is None or (len(data) < 11):
                    continue

                # data => [offset, F11, F12, F21, F22, F33, P11, P22, P33, P12, P21]
                F11, F12, F21, F22, F33 = data[1:6]
                P11, P22, P33, P12, P21 = data[6:11]

                deformation_gradient = np.eye(3)
                deformation_gradient[0, 0] = F11
                deformation_gradient[0, 1] = F12
                deformation_gradient[1, 0] = F21
                deformation_gradient[1, 1] = F22
                deformation_gradient[2, 2] = F33
                deformation_gradients_list.append(deformation_gradient)

                flattened_stress_tensor = np.array([P11, P22, P33, P12, P21])
                flattened_stresses_list.append(flattened_stress_tensor)

                stretches = np.array([F11, F22])
                stretches_list.append(stretches)

        deformation_gradients_data = torch.from_numpy(
            np.stack(deformation_gradients_list),
        ).type(torch.get_default_dtype())
        flattened_stresses_data = torch.from_numpy(
            np.stack(flattened_stresses_list),
        ).type(torch.get_default_dtype())
        stretches_data = torch.from_numpy(
            np.stack(stretches_list),
        ).type(torch.get_default_dtype())

        num_samples = len(deformation_gradients_list)
        self._validate_number_of_samples(num_samples)
        return (
            deformation_gradients_data,
            flattened_stresses_data,
            stretches_data,
        )

    def _join_input_directory_path(self) -> Path:
        return self._project_directory.get_input_subdirectory_path(
            subdir_name=self._input_directory
        )

    def _validate_file_name(self, file_name: str) -> bool:
        return (
            file_name.endswith(".m")
            and "Lambda1_" in file_name
            and "Lambda2_" in file_name
        )

    def _read_data_from_file(self, file_name: str) -> NPArray | None:
        input_path = self._join_input_file_path(file_name)
        try:
            with open(input_path, "r") as file:
                lines = file.readlines()
                data = np.array([float(line.strip()) for line in lines if line.strip()])
            return data
        except Exception as exception:
            print(f"Error reading {input_path}: {exception}")
            return None

    def _join_input_file_path(self, file_name: str) -> Path:
        return self._project_directory.get_input_file_path(
            file_name=file_name, subdir_name=self._input_directory
        )

    def _validate_number_of_samples(self, num_samples: int) -> None:
        print(f"Found {num_samples} samples in {self._input_directory}")
        if num_samples == 0:
            raise RuntimeError("No valid data loaded!")
