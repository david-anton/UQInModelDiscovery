from typing import TypeAlias

import numpy as np
import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data import (
    AllowedTestCases,
    DeformationInputs,
    StressOutputs,
    TestCases,
)
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.models.base import (
    DeformationGradient,
    Invariants,
    LSDesignMatrix,
    LSTargets,
    ParameterIndices,
    ParameterNames,
    ParameterPopulationMatrix,
    Parameters,
    PiolaStress,
    PiolaStresses,
    SplittedParameters,
    StrainEnergy,
    Stretch,
    Stretches,
    assemble_stretches_from_factors,
    assemble_stretches_from_incompressibility_assumption,
    calculate_pressure_from_incompressibility_constraint,
    count_active_parameters,
    determine_initial_parameter_mask,
    filter_active_parameter_indices,
    filter_active_parameter_names,
    init_parameter_mask,
    init_parameter_population_matrix,
    mask_and_populate_parameters,
    mask_parameters,
    update_parameter_population_matrix,
    validate_deformation_input_dimension,
    validate_input_numbers,
    validate_model_state,
    validate_parameters,
    validate_stress_output_dimension,
    validate_test_cases,
)
from bayesianmdisc.models.utility import unsqueeze_if_necessary

StretchesTuple: TypeAlias = tuple[Stretch, Stretch, Stretch]
OgdenExponents: TypeAlias = list[float]
MRExponents: TypeAlias = list[list[int]]


class IsotropicModelLibrary:

    def __init__(self, output_dim: int, device: Device):
        self._device = device
        self._degree_mr_terms = 3
        self._mr_exponents = self._determine_mr_exponents()
        self._num_regular_negative_ogden_terms = 4
        self._num_regular_positive_ogden_terms = 4
        self._min_regular_ogden_exponent = torch.tensor(-1.0, device=self._device)
        self._max_regular_ogden_exponent = torch.tensor(1.0, device=self._device)
        self._additional_ogden_terms: list[float] = []
        self._num_additional_ogden_terms = len(self._additional_ogden_terms)
        self._num_ogden_terms = self._determine_number_of_ogden_terms()
        self._ogden_exponents = self._determine_ogden_exponents()
        self._num_ln_feature_terms = 1
        self._test_case_identifier_ut = test_case_identifier_uniaxial_tension
        self._test_case_identifier_ebt = test_case_identifier_equibiaxial_tension
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ps = test_case_identifier_pure_shear
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [1, 2, 3]
        self._allowed_output_dimensions = [1, 2]
        self._zero_principal_stress_index = 2
        (
            self._num_mr_parameters,
            self._num_ogden_parameters,
            self._num_ln_feature_parameters,
        ) = self._determine_number_of_parameters()
        self._initial_num_parameters = (
            self._num_mr_parameters
            + self._num_ogden_parameters
            + self._num_ln_feature_parameters
        )
        self._initial_parameter_names = self._init_parameter_names()
        validate_stress_output_dimension(output_dim, self._allowed_output_dimensions)
        self.output_dim = output_dim
        self.num_parameters = self._initial_num_parameters
        self.parameter_names = self._initial_parameter_names
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
        self._parameter_population_matrix = init_parameter_population_matrix(
            self.num_parameters, self._device
        )

    def __call__(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        parameters: Parameters,
        validate_args=True,
    ) -> StressOutputs:
        return self.forward(inputs, test_cases, parameters, validate_args)

    def forward(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        parameters: Parameters,
        validate_args=True,
    ) -> StressOutputs:
        """The deformation input is expected to be either:
        (1) a tensor containing the stretches in all three dimensions (of shape [n, 3]) or
        (2) a tensor containing the stretches in the first and second dimension (of shape [n, 2]) or
        (3) a tensor containing only the stretch in the first dimension
            which correpsonds to the stretch factor (of shape [n, 1]).
        In case (2), the stretch in the third dimension follows directly from the assumption of
        incompressibility, regardless of the test case.
        In case (3), the stretches in the second and third dimensions are calculated under the
        assumption of perfekt incompressibility as a function of the test case."""

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        parameters = self._preprocess_parameters(parameters)
        stretches = self._assemble_stretches_if_necessary(inputs, test_cases)

        def vmap_func(stretches_: Stretches) -> PiolaStresses:
            return self._calculate_stress(stretches_, parameters)

        return vmap(vmap_func)(stretches)

    def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None:
        mask_parameters(parameter_indices, self._parameter_mask, False)

    def activate_parameters(self, parameter_indices: ParameterIndices) -> None:
        mask_parameters(parameter_indices, self._parameter_mask, True)

    def reset_parameter_deactivations(self) -> None:
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)

    def get_active_parameter_indices(self) -> ParameterIndices:
        return filter_active_parameter_indices(self._parameter_mask)

    def get_active_parameter_names(self) -> ParameterNames:
        return filter_active_parameter_names(self._parameter_mask, self.parameter_names)

    def get_number_of_active_parameters(self) -> int:
        return count_active_parameters(self._parameter_mask)

    def reduce_to_activated_parameters(self) -> None:
        old_parameter_mask = self._parameter_mask

        def reduce_num_parameters() -> None:
            self.num_parameters = self.get_number_of_active_parameters()

        def reduce_parameter_names() -> None:
            self.parameter_names = self.get_active_parameter_names()

        def reduce_parameter_mask() -> None:
            self._parameter_mask = init_parameter_mask(
                self.num_parameters, self._device
            )

        def reduce_parameter_population_matrix() -> None:
            self._parameter_population_matrix = update_parameter_population_matrix(
                self._parameter_population_matrix, old_parameter_mask
            )

        reduce_num_parameters()
        reduce_parameter_names()
        reduce_parameter_mask()
        reduce_parameter_population_matrix()

    def get_model_state(self) -> ParameterPopulationMatrix:
        return self._parameter_population_matrix

    def init_model_state(
        self, parameter_population_matrix: ParameterPopulationMatrix
    ) -> None:
        population_matrix = parameter_population_matrix
        validate_model_state(population_matrix, self._initial_num_parameters)
        initial_parameter_mask = determine_initial_parameter_mask(population_matrix)

        def init_reuced_models_num_parameters() -> None:
            self.num_parameters = population_matrix.shape[1]

        def init_reduced_models_parameter_names() -> None:
            self.parameter_names = filter_active_parameter_names(
                initial_parameter_mask, self._initial_parameter_names
            )

        def init_reduced_models_parameter_mask() -> None:
            self._parameter_mask = init_parameter_mask(
                self.num_parameters, self._device
            )

        def init_reduced_models_population_matrix() -> None:
            self._parameter_population_matrix = population_matrix

        init_reuced_models_num_parameters()
        init_reduced_models_parameter_names()
        init_reduced_models_parameter_mask()
        init_reduced_models_population_matrix()

    def set_output_dimension(self, output_dim: int) -> None:
        validate_stress_output_dimension(output_dim, self._allowed_output_dimensions)
        self.output_dim = output_dim

    def assemble_linear_system_of_equations(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        validate_args=True,
    ) -> tuple[LSDesignMatrix, LSTargets]:
        parameters = torch.ones((self.num_parameters,), device=self._device)

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        def flatten_outputs(outputs: Tensor) -> Tensor:
            if outputs.dim() == 1:
                return outputs
            else:
                return torch.transpose(outputs, 1, 0).ravel()

        def assemble_design_matrix(
            inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
        ) -> LSDesignMatrix:
            covariates = []

            for parameter_index in range(self.num_parameters):
                self._deactivate_all_parameters()
                self.activate_parameters([parameter_index])
                outputs = self.forward(inputs, test_cases, parameters)
                flattened_outputs = flatten_outputs(outputs)
                covariates += [flattened_outputs.cpu().detach().numpy()]

            self.reset_parameter_deactivations()
            return np.vstack(covariates)

        def assemble_targets(outputs: StressOutputs) -> LSTargets:
            flattened_outputs = flatten_outputs(outputs)
            return flattened_outputs.cpu().detach().numpy()

        design_matrix = assemble_design_matrix(inputs, test_cases, parameters)
        targets = assemble_targets(outputs)
        return design_matrix, targets

    def _determine_mr_exponents(self) -> MRExponents:
        exponents = []
        for n in range(1, self._degree_mr_terms + 1):
            exponents_cI_1 = torch.linspace(
                start=0, end=n, steps=n + 1, dtype=torch.int64
            )
            exponents_cI_1 = exponents_cI_1.reshape((-1, 1))
            exponents_cI_2 = torch.flip(exponents_cI_1, dims=(0,))
            exponents += [torch.concat((exponents_cI_1, exponents_cI_2), dim=1)]
        return torch.concat(exponents, dim=0).tolist()

    def _determine_number_of_ogden_terms(self) -> int:
        return (
            self._num_regular_negative_ogden_terms
            + self._num_regular_positive_ogden_terms
            + self._num_additional_ogden_terms
        )

    def _determine_ogden_exponents(self) -> OgdenExponents:
        negative_exponents = torch.linspace(
            start=self._min_regular_ogden_exponent,
            end=0.0,
            steps=self._num_regular_negative_ogden_terms + 1,
        )[:-1].tolist()
        positive_exponents = torch.linspace(
            start=0.0,
            end=self._max_regular_ogden_exponent,
            steps=self._num_regular_positive_ogden_terms + 1,
        )[1:].tolist()
        return negative_exponents + positive_exponents + self._additional_ogden_terms

    def _determine_allowed_test_cases(self) -> AllowedTestCases:
        return torch.tensor(
            [
                self._test_case_identifier_ut,
                self._test_case_identifier_ebt,
                self._test_case_identifier_bt,
                self._test_case_identifier_ps,
            ],
            device=self._device,
        )

    def _determine_number_of_parameters(
        self,
    ) -> tuple[int, int, int]:

        def determine_number_of_mr_parameters() -> int:
            num_parameters = 0
            for n in range(1, self._degree_mr_terms + 1):
                num_parameters += n + 1
            return num_parameters

        def determine_number_of_ogden_parameters() -> int:
            return self._num_ogden_terms

        def determine_number_of_ln_feature_parameters() -> int:
            return self._num_ln_feature_terms

        num_mr_parameters = determine_number_of_mr_parameters()
        num_ogden_parameters = determine_number_of_ogden_parameters()
        num_ln_feature_parameters = determine_number_of_ln_feature_parameters()
        return num_mr_parameters, num_ogden_parameters, num_ln_feature_parameters

    def _init_parameter_names(self) -> ParameterNames:

        def compose_mr_parameter_names() -> ParameterNames:
            parameter_names = []
            for n in range(1, self._degree_mr_terms + 1):
                for m in range(n + 1):
                    exponent_I_1 = m
                    exponent_I_2 = n - m
                    parameter_name = f"C_{exponent_I_1}_{exponent_I_2}"
                    if exponent_I_1 == 1 and exponent_I_2 == 0:
                        parameter_name = parameter_name + " (NH)"
                    elif exponent_I_1 == 0 and exponent_I_2 == 1:
                        parameter_name = parameter_name + " (MR)"
                    parameter_names += [parameter_name]
            return tuple(parameter_names)

        def compose_ogden_parameter_names() -> ParameterNames:
            parameter_names = []
            for exponent in self._ogden_exponents:
                parameter_names += [f"Ogden ({round(exponent,3)})"]
            return tuple(parameter_names)

        def compose_ln_feature_parameter_name() -> ParameterNames:
            return ("ln_I2",)

        mr_parameter_names = compose_mr_parameter_names()
        ogden_parameter_names = compose_ogden_parameter_names()
        ln_feature_parameter_name = compose_ln_feature_parameter_name()
        return mr_parameter_names + ogden_parameter_names + ln_feature_parameter_name

    def _validate_inputs(
        self, inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
    ) -> None:
        validate_input_numbers(inputs, test_cases)
        validate_deformation_input_dimension(inputs, self._allowed_input_dimensions)
        validate_test_cases(test_cases, self._allowed_test_cases)
        validate_parameters(parameters, self.num_parameters)

    def _preprocess_parameters(self, parameters: Parameters) -> Parameters:
        return mask_and_populate_parameters(
            parameters, self._parameter_mask, self._parameter_population_matrix
        )

    def _assemble_stretches_if_necessary(
        self, stretches: Stretches, test_cases: TestCases
    ):
        stretch_dim = stretches.shape[1]
        if stretch_dim == 1:
            return assemble_stretches_from_factors(stretches, test_cases, self._device)
        if stretch_dim == 2:
            return assemble_stretches_from_incompressibility_assumption(
                stretches, self._device
            )
        else:
            return stretches

    def _calculate_stress(
        self, stretches: Stretches, parameters: Parameters
    ) -> PiolaStress:
        F = self._assemble_deformation_gradient(stretches)
        F_inverse_transpose = F.inverse().transpose(0, 1)
        dW_dF = grad(self._calculate_strain_energy, argnums=0)(F, parameters)
        p = calculate_pressure_from_incompressibility_constraint(
            F, dW_dF, self._zero_principal_stress_index
        )

        P = dW_dF - p * F_inverse_transpose

        P11 = P[0, 0]
        P22 = P[1, 1]
        if self.output_dim == 1:
            return unsqueeze_if_necessary(P11)
        else:
            return torch.concat(
                (unsqueeze_if_necessary(P11), unsqueeze_if_necessary(P22))
            )

    def _assemble_deformation_gradient(
        self, stretches: Stretches
    ) -> DeformationGradient:
        zero = torch.tensor([0.0], device=self._device)
        F_11, F_22, F_33 = self._split_stretches(stretches)
        row_1 = torch.concat((self._unsqueeze_zero_dimension(F_11), zero, zero))
        row_2 = torch.concat((zero, self._unsqueeze_zero_dimension(F_22), zero))
        row_3 = torch.concat((zero, zero, self._unsqueeze_zero_dimension(F_33)))
        return torch.stack((row_1, row_2, row_3), dim=0)

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        mr_parameters, ogden_parameters, ln_feature_parameters = self._split_parameters(
            parameters
        )
        mr_strain_energy_terms = self._calculate_mr_strain_energy_terms(
            deformation_gradient, mr_parameters
        )
        ogden_strain_energy_terms = self._calculate_ogden_strain_energy_terms(
            deformation_gradient, ogden_parameters
        )
        ln_feature_strain_energy_term = self._calculate_ln_feature_strain_energy_terms(
            deformation_gradient, ln_feature_parameters
        )
        return (
            mr_strain_energy_terms
            + ogden_strain_energy_terms
            + ln_feature_strain_energy_term
        )

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        return torch.split(
            parameters,
            [
                self._num_mr_parameters,
                self._num_ogden_parameters,
                self._num_ln_feature_parameters,
            ],
        )

    def _calculate_mr_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        cI_1, cI_2 = self._calculate_corrected_invariants(deformation_gradient)

        terms = torch.concat(
            [
                self._unsqueeze_zero_dimension(
                    cI_1 ** exponents[0] * cI_2 ** exponents[1]
                )
                for exponents in self._mr_exponents
            ]
        )
        weighted_terms = parameters * terms
        return torch.sum(weighted_terms)

    def _calculate_ogden_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        one = torch.tensor(1.0, device=self._device)
        three = torch.tensor(3.0, device=self._device)
        stretches = self._extract_stretches(deformation_gradient)

        terms = torch.concat(
            [
                (torch.sum(stretches**exponent, dim=0, keepdim=True) - three)
                for exponent in self._ogden_exponents
            ]
        )

        weighted_terms = parameters * terms
        return torch.sum(weighted_terms)

    def _calculate_ln_feature_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        _, I_2 = self._calculate_invariants(deformation_gradient)
        three = torch.tensor(3.0, device=self._device)
        ln_feature_parameter = parameters[0]
        return ln_feature_parameter * torch.log(I_2 / three)

    def _calculate_corrected_invariants(
        self, deformation_gradient: DeformationGradient
    ) -> Invariants:
        # Invariants
        I_1, I_2 = self._calculate_invariants(deformation_gradient)
        # Constants
        three = torch.tensor(3.0, device=self._device)
        # Corrected invariants
        I_1_cor = I_1 - three
        I_2_cor = I_2 - three
        return I_1_cor, I_2_cor

    def _calculate_invariants(
        self, deformation_gradient: DeformationGradient
    ) -> Invariants:
        # Deformation tensors
        F = deformation_gradient
        C = torch.matmul(F.transpose(0, 1), F)  # right Cauchy-Green deformation tensor
        # Constants
        half = torch.tensor(1 / 2, device=self._device)
        # Isotropic invariants
        I_1 = torch.trace(C)
        I_2 = half * (I_1**2 - torch.tensordot(C, C))
        return I_1, I_2

    def _split_stretches(self, stretches: Stretches) -> StretchesTuple:
        F_11 = stretches[0]
        F_22 = stretches[1]
        F_33 = stretches[2]
        return F_11, F_22, F_33

    def _extract_stretches(
        self, deformation_gradient: DeformationGradient
    ) -> Stretches:
        return torch.diag(deformation_gradient)

    def _unsqueeze_zero_dimension(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, dim=0)

    def _deactivate_all_parameters(self) -> None:
        parameter_indices = list(range(self.num_parameters))
        self.deactivate_parameters(parameter_indices)
