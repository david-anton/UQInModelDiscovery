from typing import Protocol, TypeAlias

import torch

from bayesianmdisc.customtypes import Tensor
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
StrainEnergyGradient: TypeAlias = Tensor
StrainEnergyGradients: TypeAlias = tuple[StrainEnergyGradient, ...]
IncompressibilityConstraint: TypeAlias = Tensor
Parameters: TypeAlias = Tensor
SplittedParameters: TypeAlias = tuple[Parameters, ...]
ParameterNames: TypeAlias = tuple[str, ...]
TrueParameters: TypeAlias = tuple[float, ...]


class ModelProtocol(Protocol):
    output_dim: int
    num_parameters: int

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
