from typing import TypeAlias

import numpy as np
import pandas as pd
import torch

from bayesianmdisc.customtypes import Device, NPArray, PDDataFrame
from bayesianmdisc.data.base import (
    Data,
    DeformationInputs,
    TestCases,
    assemble_test_case_identifiers,
    convert_to_torch,
    flatten_and_stack_arrays,
    numpy_data_type,
    stack_arrays,
)
from bayesianmdisc.errors import DataError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import OrthotropicCANN
from bayesianmdisc.testcases import (
    TestCaseIdentifier,
    map_test_case_identifiers_to_labels,
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
)

StrainComponent: TypeAlias = tuple[int, int]
StressIndex: TypeAlias = int
StretchRatio: TypeAlias = tuple[float, float]

min_shear_strain = 0.0
max_shear_strain = 0.5
min_nominal_principel_stretch = 1.0
max_nominal_principal_stretch = 1.1
irrelevant_stress_components = [4]

test_case_identifiers_ss = [
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_32,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_23,
]
stretch_ratios = [
    (1.0, 1.0),
    (1.0, 0.75),
    (0.75, 1.0),
    (1.0, 0.5),
    (0.5, 1.0),
]

row_offset = 3
excel_sheet_name = "Sheet1"


class LinkaHeartDataSet:
    def __init__(
        self,
        file_name: str,
        input_directory: str,
        project_directory: ProjectDirectory,
        device: Device,
        consider_shear_data: bool = True,
        consider_extension_data: bool = True,
    ):
        self._validate_data_configuration(consider_shear_data, consider_extension_data)
        self._consider_shear_data = consider_shear_data
        self._consider_biaxial_data = consider_extension_data
        self._file_name = file_name
        self._input_directory = input_directory
        self._project_directory = project_directory
        self._device = device
        self._start_column_shear = 0
        self._column_offsets_shear = [2, 3, 2, 3, 2]
        self._start_column_biaxial = 15
        self._num_data_sets_biaxial = 5
        self._column_offsets_biaxial = [5, 5, 5, 5]
        self._np_data_type = numpy_data_type
        self._data_frame = self._init_data_frame()

    def read_data(self) -> Data:
        all_deformation_gradients = []
        all_test_cases = []
        all_stresse_tensors = []

        if self._consider_shear_data:
            column = self._start_column_shear

            for test_case_index, test_case_identifier in enumerate(
                test_case_identifiers_ss
            ):
                deformations_column, test_cases_colum, stresses_column = (
                    self._read_shear_data(column, test_case_identifier)
                )
                all_deformation_gradients.append(deformations_column)
                all_test_cases.append(test_cases_colum)
                all_stresse_tensors.append(stresses_column)
                if test_case_index < len(test_case_identifiers_ss) - 1:
                    column += self._column_offsets_shear[test_case_index]

        if self._consider_biaxial_data:
            column = self._start_column_biaxial

            for test_case_index in range(self._num_data_sets_biaxial):
                deformations_column, test_cases_colum, stresses_column = (
                    self._read_biaxial_data(column)
                )
                all_deformation_gradients.append(deformations_column)
                all_test_cases.append(test_cases_colum)
                all_stresse_tensors.append(stresses_column)
                if test_case_index < self._num_data_sets_biaxial - 1:
                    column += self._column_offsets_biaxial[test_case_index]

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

    def generate_uniform_inputs(
        self, num_points_per_test_case: int
    ) -> tuple[DeformationInputs, TestCases]:
        all_deformation_gradients = []
        all_test_cases = []

        if self._consider_shear_data:
            shear_strains = generate_shear_strains(num_points_per_test_case)

            for test_case_identifier in test_case_identifiers_ss:
                all_deformation_gradients += [
                    assemble_flattened_deformation_gradients(
                        shear_strains, test_case_identifier
                    )
                ]
                all_test_cases += [
                    assemble_test_case_identifiers(test_case_identifier, shear_strains)
                ]

        if self._consider_biaxial_data:
            for stretch_ratio in stretch_ratios:
                test_case_identifier = test_case_identifier_biaxial_tension
                principal_stretches = generate_principal_stretches(
                    stretch_ratio, num_points_per_test_case
                )
                all_deformation_gradients += [
                    assemble_flattened_deformation_gradients(
                        principal_stretches, test_case_identifier
                    )
                ]
                all_test_cases += [
                    assemble_test_case_identifiers(
                        test_case_identifier, principal_stretches
                    )
                ]

        deformation_gradients = stack_arrays(all_deformation_gradients)
        test_cases = stack_arrays(all_test_cases)
        test_cases = test_cases.reshape((-1,))

        deformation_gradients_torch = convert_to_torch(
            deformation_gradients, self._device
        )
        test_cases_torch = convert_to_torch(test_cases, self._device)
        return deformation_gradients_torch, test_cases_torch

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
        return pd.read_excel(input_path, sheet_name=excel_sheet_name)

    def _read_shear_data(
        self, start_column: int, test_case_identifier: TestCaseIdentifier
    ) -> tuple[NPArray, NPArray, NPArray]:
        shear_strains = self._read_column(start_column).reshape((-1, 1))
        shear_stresses = self._read_column(start_column + 1).reshape((-1, 1))

        flattened_deformation_gradients = assemble_flattened_deformation_gradients(
            shear_strains, test_case_identifier
        )
        test_cases = assemble_test_case_identifiers(
            test_case_identifier, flattened_deformation_gradients
        )
        reduced_flattened_stress_tensors = assemble_reduced_flattened_stress_tensor(
            shear_stresses, test_case_identifier
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
            stretches, test_case_identifier_biaxial_tension
        )
        test_cases = assemble_test_case_identifiers(
            test_case_identifier_biaxial_tension, flattened_deformation_gradients
        )
        reduced_flattened_stress_tensors = assemble_reduced_flattened_stress_tensor(
            stresses, test_case_identifier_biaxial_tension
        )
        return (
            flattened_deformation_gradients,
            test_cases,
            reduced_flattened_stress_tensors,
        )

    def _read_column(self, column: int) -> NPArray:
        return (
            self._data_frame.iloc[row_offset:, column]
            .dropna()
            .astype(self._np_data_type)
            .values
        )


class LinkaHeartDataSetGenerator:

    def __init__(
        self,
        model: OrthotropicCANN,
        parameters: tuple[float, ...],
        num_point_per_test_case: int,
        file_name: str,
        output_directory: str,
        project_directory: ProjectDirectory,
        device: Device,
    ) -> None:
        self._model = model
        self._parameters = parameters
        self._num_points_per_test_case = num_point_per_test_case
        self._file_name = file_name
        self._output_directory = output_directory
        self._project_directory = project_directory
        self._device = device
        self._start_column_indices_ss = [0, 2, 5, 7, 10, 12]
        self._index_fiber = 0
        self._index_normal = 1
        self._start_column_indices_bt = [15, 20, 25, 30, 35]

    def generate(self) -> None:
        data_frame = self._init_data_frame()
        self._generate_shear_data(data_frame)
        self._generate_biaxial_data(data_frame)
        self._write_data_frame(data_frame)

    def _init_data_frame(self) -> PDDataFrame:
        return pd.DataFrame()

    def _generate_shear_data(self, data_frame: PDDataFrame) -> None:
        shear_strains = generate_shear_strains(self._num_points_per_test_case)

        for test_case_index in range(len(test_case_identifiers_ss)):
            start_column_index = self._start_column_indices_ss[test_case_index]
            self._add_shear_data_to_data_frame(
                shear_strains=shear_strains,
                test_case_identifier=test_case_identifiers_ss[test_case_index],
                data_frame=data_frame,
                start_column_index=start_column_index,
            )
            if test_case_index != 0 and test_case_index % 2 != 0:
                self._add_empty_column(data_frame, start_column_index + 2)

    def _generate_biaxial_data(self, data_frame: PDDataFrame) -> None:

        for test_case_index in range(len(stretch_ratios)):
            stretch_ratio = stretch_ratios[test_case_index]
            stretches = generate_principal_stretches(
                stretch_ratio, self._num_points_per_test_case
            )
            start_column_index = self._start_column_indices_bt[test_case_index]
            self._add_biaxial_data_to_data_frame(
                stretches=stretches,
                test_case_identifier=test_case_identifier_biaxial_tension,
                stretch_ratio=stretch_ratio,
                data_frame=data_frame,
                start_column_index=start_column_index,
            )
            if not test_case_index == len(stretch_ratios) - 1:
                self._add_empty_column(data_frame, start_column_index + 4)

    def _add_shear_data_to_data_frame(
        self,
        shear_strains: NPArray,
        test_case_identifier: TestCaseIdentifier,
        data_frame: PDDataFrame,
        start_column_index: int,
    ) -> None:

        def add_shear_data(
            shear_strains: NPArray,
            shear_stresses: NPArray,
            test_case_identifier: TestCaseIdentifier,
            data_frame: PDDataFrame,
            start_column_index: int,
        ) -> None:
            column_index_strains = start_column_index
            column_index_stresses = start_column_index + 1

            test_case_label = map_test_case_identifiers_to_labels(
                torch.tensor([test_case_identifier])
            )[0]
            label_strains = test_case_label + " - gamma [-]"
            label_stresses = test_case_label + " - sigma [kPa]"
            shear_strains = shear_strains.reshape((-1,))
            self._insert_one_column(
                data_frame, column_index_strains, label_strains, shear_strains
            )
            self._insert_one_column(
                data_frame, column_index_stresses, label_stresses, shear_stresses
            )

        full_outputs = self._forward_model(shear_strains, test_case_identifier)
        output_index = _map_to_reduced_shear_stess_index(test_case_identifier)
        shear_stresses = full_outputs[:, output_index]
        add_shear_data(
            shear_strains=shear_strains,
            shear_stresses=shear_stresses,
            test_case_identifier=test_case_identifier,
            data_frame=data_frame,
            start_column_index=start_column_index,
        )

    def _add_biaxial_data_to_data_frame(
        self,
        stretches: NPArray,
        test_case_identifier: TestCaseIdentifier,
        stretch_ratio: StretchRatio,
        data_frame: PDDataFrame,
        start_column_index: int,
    ) -> None:

        def add_biaxial_data(
            stretches: NPArray,
            stresses: NPArray,
            test_case_identifier: TestCaseIdentifier,
            stretch_ratio: StretchRatio,
            data_frame: PDDataFrame,
            start_column_index: int,
        ) -> None:
            column_index_stretch_f = start_column_index
            column_index_stress_ff = start_column_index + 1
            column_index_stretch_n = start_column_index + 2
            column_index_stress_nn = start_column_index + 3

            stretches_f = stretches[:, self._index_fiber]
            stresses_ff = stresses[:, self._index_fiber]
            stretches_n = stretches[:, self._index_normal]
            stresses_nn = stresses[:, self._index_normal]

            test_case_label = map_test_case_identifiers_to_labels(
                torch.tensor([test_case_identifier])
            )[0]
            stretch_ratio_label = f" - ratio {stretch_ratio[0]}:{stretch_ratio[1]}"
            label_stretch_f = test_case_label + stretch_ratio_label + " - lambda f [-]"
            label_stress_ff = (
                test_case_label + stretch_ratio_label + " - sigma_ff [kPa]"
            )
            label_stretch_n = test_case_label + stretch_ratio_label + " - lambda n [-]"

            label_stress_nn = (
                test_case_label + stretch_ratio_label + " - sigma_nn [kPa]"
            )

            self._insert_one_column(
                data_frame, column_index_stretch_f, label_stretch_f, stretches_f
            )
            self._insert_one_column(
                data_frame, column_index_stress_ff, label_stress_ff, stresses_ff
            )
            self._insert_one_column(
                data_frame, column_index_stretch_n, label_stretch_n, stretches_n
            )
            self._insert_one_column(
                data_frame, column_index_stress_nn, label_stress_nn, stresses_nn
            )

        full_outputs = self._forward_model(stretches, test_case_identifier)
        stresses_ff = full_outputs[:, 0].reshape((-1, 1))
        stresses_nn = full_outputs[:, 7].reshape((-1, 1))
        stresses = np.hstack((stresses_ff, stresses_nn))
        add_biaxial_data(
            stretches=stretches,
            stresses=stresses,
            test_case_identifier=test_case_identifier,
            stretch_ratio=stretch_ratio,
            data_frame=data_frame,
            start_column_index=start_column_index,
        )

    def _forward_model(
        self, deformation_inputs: NPArray, test_case_identifier: TestCaseIdentifier
    ) -> NPArray:
        deformation_gradients = assemble_flattened_deformation_gradients(
            deformation_inputs, test_case_identifier
        )
        inputs = (
            torch.from_numpy(deformation_gradients)
            .type(torch.get_default_dtype())
            .to(self._device)
        )
        test_cases = torch.full(
            (len(inputs),), test_case_identifier, dtype=torch.int, device=self._device
        )
        parameters = torch.tensor(self._parameters, device=self._device)
        outputs = self._model(inputs, test_cases, parameters)
        return outputs.detach().cpu().numpy()

    def _add_empty_column(self, data_frame: PDDataFrame, column_index: int) -> None:
        self._insert_one_column(
            data_frame,
            column_index,
            column_label=f"empty_{column_index}",
            column_values=np.full((self._num_points_per_test_case,), np.nan),
        )

    def _insert_one_column(
        self,
        data_frame: PDDataFrame,
        column_index: int,
        column_label: str,
        column_values: NPArray,
    ) -> None:
        data_frame.insert(column_index, column_label, pd.Series(column_values))

    def _write_data_frame(self, data_frame: PDDataFrame) -> None:
        output_path = self._project_directory.create_input_file_path(
            file_name=self._file_name, subdir_name=self._output_directory
        )
        data_frame.to_excel(
            output_path, sheet_name=excel_sheet_name, index=False, startrow=row_offset
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
        shear_strain = deformation_input[0]
        shear_component = _map_to_shear_strain_components(test_case_identifier)
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
        shear_stress = stress_output[0]
        shear_component = _map_to_shear_strain_components(test_case_identifier)
        symmetric_shear_component = tuple(reversed(shear_component))
        stress_tensor = np.zeros((3, 3), dtype=numpy_data_type)
        stress_tensor[shear_component] = shear_stress
        stress_tensor[symmetric_shear_component] = shear_stress
        return stress_tensor

    def _reduce_to_relevant_stresses(flattened_stress_tensors: NPArray) -> NPArray:
        return np.delete(flattened_stress_tensors, irrelevant_stress_components, axis=1)

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


def generate_shear_strains(num_points: int) -> NPArray:
    return np.linspace(min_shear_strain, max_shear_strain, num=num_points).reshape(
        (-1, 1)
    )


def generate_principal_stretches(
    stretch_ratio: tuple[float, float], num_points: int
) -> NPArray:
    nominal_principal_stretches = np.linspace(
        min_nominal_principel_stretch, max_nominal_principal_stretch, num=num_points
    ).reshape((-1, 1))
    ones = np.ones_like(nominal_principal_stretches)
    ratio = np.array(stretch_ratio)
    return ones + ratio * (nominal_principal_stretches - ones)


def _map_to_shear_strain_components(
    shear_test_case_identifier: TestCaseIdentifier,
) -> StrainComponent:
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


def _map_to_reduced_shear_stess_index(
    shear_test_case_identifier: TestCaseIdentifier,
) -> StressIndex:
    if shear_test_case_identifier == test_case_identifier_simple_shear_12:
        return 3
    elif shear_test_case_identifier == test_case_identifier_simple_shear_21:
        return 1
    elif shear_test_case_identifier == test_case_identifier_simple_shear_13:
        return 5
    elif shear_test_case_identifier == test_case_identifier_simple_shear_31:
        return 2
    elif shear_test_case_identifier == test_case_identifier_simple_shear_23:
        return 6
    elif shear_test_case_identifier == test_case_identifier_simple_shear_32:
        return 4
    else:
        raise DataError(f"Unvalid test case identifier: {shear_test_case_identifier}")
