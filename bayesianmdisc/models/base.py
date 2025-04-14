from itertools import compress
from typing import Protocol, TypeAlias

import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.data import DeformationInputs, StressOutputs, TestCases
from bayesianmdisc.data.testcases import AllowedTestCases
from bayesianmdisc.errors import ModelError

Stretch: TypeAlias = Tensor
Stretches: TypeAlias = Tensor
DeformationGradient: TypeAlias = Tensor
Invariant: TypeAlias = Tensor
Invariants: TypeAlias = tuple[Invariant, ...]
CauchyStress: TypeAlias = Tensor
CauchyStresses: TypeAlias = Tensor
PiolaStress: TypeAlias = Tensor
PiolaStresses: TypeAlias = Tensor
StrainEnergy: TypeAlias = Tensor
StrainEnergyDerivatives: TypeAlias = Tensor
StrainEnergyDerivative: TypeAlias = Tensor
StrainEnergyDerivativesTuple: TypeAlias = tuple[StrainEnergyDerivative, ...]
IncompressibilityConstraint: TypeAlias = Tensor
Parameters: TypeAlias = Tensor
SplittedParameters: TypeAlias = tuple[Parameters, ...]
ParameterNames: TypeAlias = tuple[str, ...]
TrueParameters: TypeAlias = tuple[float, ...]
ParameterMask: TypeAlias = Tensor
ParameterIndices: TypeAlias = list[int]
ParameterPopulationMatrix = Tensor


class ModelProtocol(Protocol):
    output_dim: int
    num_parameters: int
    parameter_names: ParameterNames

    def __call__(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        parameters: Parameters,
        validate_args: bool = True,
    ) -> StressOutputs:
        pass

    def forward(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        parameters: Parameters,
        validate_args: bool = True,
    ) -> StressOutputs:
        pass

    def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None: ...

    def activate_parameters(self, parameter_indices: ParameterIndices) -> None: ...

    def reset_parameter_deactivations(self) -> None: ...

    def get_active_parameter_names(self) -> ParameterNames: ...

    def get_number_of_active_parameters(self) -> int: ...

    def reduce_to_activated_parameters(self) -> None: ...

    def get_model_state(self) -> ParameterPopulationMatrix: ...

    def init_model_state(
        self, parameter_population_indices: ParameterPopulationMatrix
    ) -> None: ...


def validate_input_numbers(inputs: DeformationInputs, test_cases: TestCases) -> None:
    num_inputs = len(inputs)
    num_test_cases = len(test_cases)
    if num_inputs != num_test_cases:
        raise ModelError(
            f"""The number of inputs and test cases is expected to be the same,
            but is {num_inputs} and {num_test_cases}."""
        )


def validate_deformation_input_dimension(
    inputs: DeformationInputs, allowed_dimensions: list[int]
) -> None:
    input_dimension = inputs.shape[1]
    if not input_dimension in allowed_dimensions:
        raise ModelError(
            f"""The dimension of deformation inputs is {input_dimension} 
            which is not allowed."""
        )


def validate_test_cases(
    test_cases: TestCases, allowed_test_cases: AllowedTestCases
) -> None:
    for test_case in test_cases:
        if not test_case in allowed_test_cases:
            raise ModelError(
                f"""The list of test cases contains an unvalid test case {test_case}."""
            )


def validate_parameters(parameters: Parameters, expected_num_parameters: int) -> None:
    parameter_size = parameters.size()
    expected_size = torch.Size([expected_num_parameters])
    if not parameter_size == expected_size:
        raise ModelError(
            f"""The size of parameters is expected to be {expected_size}, 
            but is {parameter_size}"""
        )


def init_parameter_mask(num_parameters: int, device: Device) -> ParameterMask:
    return torch.full((num_parameters,), True, device=device)


def init_parameter_population_matrix(
    num_parameters: int, device: Device
) -> ParameterPopulationMatrix:
    return torch.eye(num_parameters, dtype=torch.int64, device=device)


def update_parameter_population_matrix(
    population_matrix: ParameterPopulationMatrix,
    parameter_mask: ParameterMask,
) -> ParameterPopulationMatrix:
    num_columns = len(parameter_mask)
    num_deleted_columns = 0
    for column in range(num_columns):
        if not parameter_mask[column]:
            corrected_column = column - num_deleted_columns
            population_matrix = torch.concat(
                (
                    population_matrix[:, :corrected_column],
                    population_matrix[:, corrected_column:],
                ),
                dim=1,
            )
            num_deleted_columns += 1
    return population_matrix


def mask_parameters(
    parameter_indices: ParameterIndices, parameter_mask: ParameterMask, mask_value: bool
) -> None:
    for indice in parameter_indices:
        parameter_mask[indice] = mask_value


def count_active_parameters(parameter_mask: ParameterMask) -> int:
    return int(torch.sum(parameter_mask))


def filter_active_parameter_names(
    parameter_mask: ParameterMask, parameter_names: ParameterNames
) -> ParameterNames:
    parameter_mask_list = parameter_mask.detach().cpu().tolist()
    return tuple(compress(parameter_names, parameter_mask_list))


def preprocess_parameters(
    parameters: Parameters,
    parameter_mask: ParameterMask,
    parameter_population_matrix: ParameterPopulationMatrix,
) -> Parameters:
    masked_parameters = parameter_mask * parameters
    return torch.matmul(parameter_population_matrix, masked_parameters)


def populate_parameters(
    parameters: Parameters, parameter_population_matrix: ParameterPopulationMatrix
) -> Parameters:
    return torch.matmul(parameter_population_matrix, parameters)


def validate_model_state(
    parameter_population_matrix: ParameterPopulationMatrix, num_initial_parameters: int
) -> None:
    def validate_dimensions() -> None:
        dims = parameter_population_matrix.dim()
        expected_dims = 2
        if not dims == expected_dims:
            raise ModelError(
                f"""The population matrix is expected to be two-dimensional."""
            )

    def validate_number_of_columns() -> None:
        num_columns = parameter_population_matrix.shape[0]
        expected_num_columns = num_initial_parameters
        if not num_columns == expected_num_columns:
            raise ModelError(
                f"""The number of columns of the population matrix is expected 
                to match the number of initial model parameters."""
            )

    validate_dimensions()
    validate_number_of_columns()


def determine_initial_parameter_mask(
    parameter_population_matrix: ParameterPopulationMatrix,
) -> ParameterMask:
    return torch.greater(torch.sum(parameter_population_matrix, dim=1), 0)
