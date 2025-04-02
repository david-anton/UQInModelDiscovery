from typing import TypeAlias

import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.data import (
    AllowedTestCases,
    DeformationInputs,
    StressOutputs,
    TestCases,
)
from bayesianmdisc.data.testcases import (
    TestCase,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.models.base import (
    DeformationGradient,
    IncompressibilityConstraint,
    Invariant,
    Invariants,
    ParameterNames,
    Parameters,
    PiolaStress,
    PiolaStresses,
    SplittedParameters,
    StrainEnergy,
    StrainEnergyGradient,
    Stretch,
    Stretches,
    validate_deformation_input_dimension,
    validate_input_numbers,
    validate_parameters,
    validate_test_cases,
)

StretchesTuple: TypeAlias = tuple[Stretch, Stretch, Stretch]


class TreloarCANN:

    def __init__(self, device: Device):
        self._device = device
        self._num_cann_invariants = 2
        self._num_cann_invariant_power_terms = 2
        self._num_cann_activation_functions = 3
        self._num_cann_parameters_per_invariant_power_term = 5
        self._cann_exponential_weight_scale = torch.tensor(1e-4, device=self._device)
        self._num_ogden_terms = 20
        self._min_ogden_exponent = torch.tensor(-5.0, device=self._device)
        self._max_ogden_exponent = torch.tensor(5.0, device=self._device)
        self._ogden_exponents = self._determine_ogden_exponents()
        self._test_case_identifier_ut = test_case_identifier_uniaxial_tension
        self._test_case_identifier_ebt = test_case_identifier_equibiaxial_tension
        self._test_case_identifier_ps = test_case_identifier_pure_shear
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [1, 3]
        self._num_cann_parameters, self._num_ogden_parameters = (
            self._determine_number_of_parameters()
        )
        self.output_dim = 1
        self.num_parameters = self._num_cann_parameters + self._num_ogden_parameters

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
        (1) a tensor containing the stretches in all three dimensions or
        (2) a tensor containing only the stretch in the first dimension
            which correpsonds to the stretch factor.
        In case (2), the stretches in the second and third dimensions are calculated
        from the stretch factor depending on the test case."""

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        stretches = self._assemble_stretches_if_necessary(inputs)

        def vmap_func(stretches_: Stretches) -> PiolaStresses:
            return self._calculate_stress(stretches_, parameters)

        return vmap(vmap_func)(stretches, test_cases)

    def get_parameter_names(self) -> ParameterNames:

        def compose_cann_parameter_names() -> ParameterNames:
            parameter_names = []
            first_layer_indices = 2
            second_layer_indices = 1
            invariant_names = ["I_1", "I_2"]
            power_names = ["p1", "p2"]
            activation_function_names = ["i", "exp", "ln"]

            for invariant in invariant_names:
                for power in power_names:
                    # first layer
                    for activation_function in activation_function_names[1:]:
                        parameter_names += [
                            f"W_1_{first_layer_indices} (l1, {invariant}, {power}, {activation_function})"
                        ]
                        first_layer_indices += 1
                    first_layer_indices += 1

                    # second layer
                    for activation_function in activation_function_names:
                        parameter_names += [
                            f"W_2_{second_layer_indices} (l2, {invariant}, {power}, {activation_function})"
                        ]
                        second_layer_indices += 1
            return tuple(parameter_names)

        def compose_ogden_parameter_names() -> ParameterNames:
            parameter_names = []
            for index, exponent in zip(
                range(1, self._num_ogden_terms + 1), self._ogden_exponents.tolist()
            ):
                parameter_names += [f"O_{index} (exponent: {round(exponent,2)})"]
            return tuple(parameter_names)

        cann_parameter_names = compose_cann_parameter_names()
        ogden_parameter_names = compose_ogden_parameter_names()
        return cann_parameter_names + ogden_parameter_names

    def _determine_number_of_parameters(self) -> tuple[int, int]:
        def determine_number_of_cann_parameters() -> int:
            num_invariants = self._num_cann_invariants
            num_power_terms = self._num_cann_invariant_power_terms
            num_activation_functions = self._num_cann_activation_functions
            num_parameters_first_layer = (
                num_invariants * num_power_terms * (num_activation_functions - 1)
            )
            num_parameters_second_layer = (
                num_invariants * num_power_terms * num_activation_functions
            )
            return num_parameters_first_layer + num_parameters_second_layer

        def determine_number_of_ogden_parameters() -> int:
            return self._num_ogden_terms

        num_cann_parameters = determine_number_of_cann_parameters()
        num_ogden_parameters = determine_number_of_ogden_parameters()
        return num_cann_parameters, num_ogden_parameters

    def _determine_ogden_exponents(self) -> Tensor:
        return torch.linspace(
            start=self._min_ogden_exponent,
            end=self._max_ogden_exponent,
            steps=self._num_ogden_terms,
        )

    def _determine_allowed_test_cases(self) -> AllowedTestCases:
        return torch.tensor(
            [
                self._test_case_identifier_ut,
                self._test_case_identifier_ebt,
                self._test_case_identifier_ps,
            ],
            device=self._device,
        )

    def _validate_inputs(
        self, inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
    ) -> None:
        validate_input_numbers(inputs, test_cases)
        validate_deformation_input_dimension(inputs, self._allowed_input_dimensions)
        validate_test_cases(test_cases, self._allowed_test_cases)
        validate_parameters(parameters, self.num_parameters)

    def _assemble_stretches_if_necessary(
        self, stretches: Stretches, test_cases: TestCases
    ):
        if stretches.dim() == 1:
            stretch_facors = stretches
            return self._assemble_stretches_from_factors(stretch_facors, test_cases)
        else:
            return stretches

    def _assemble_stretches_from_factors(
        self, stretch_factors: Stretches, test_cases: TestCases
    ):
        indices_ut = test_cases == self._test_case_identifier_ut
        indices_ebt = test_cases == self._test_case_identifier_ebt
        indices_ps = test_cases == self._test_case_identifier_ps

        stretch_factors_ut = stretch_factors[indices_ut]
        stretch_factors_ebt = stretch_factors[indices_ebt]
        stretch_factors_ps = stretch_factors[indices_ps]

        one = torch.tensor(1.0, device=self._device)

        def calculate_stretches_ut(stretch_factors: Stretches) -> Stretches:
            stretch_1 = stretch_factors
            stretch_2 = stretch_3 = one / torch.sqrt(stretch_factors)
            return torch.concat((stretch_1, stretch_2, stretch_3), dim=1)

        def calculate_stretches_ebt(stretch_factors: Stretches) -> Stretches:
            stretch_1 = stretch_2 = stretch_factors
            stretch_3 = one / stretch_factors**2
            return torch.concat((stretch_1, stretch_2, stretch_3), dim=1)

        def calculate_stretches_ps(stretch_factors: Stretches) -> Stretches:
            stretch_1 = stretch_factors
            stretch_2 = torch.ones_like(stretch_factors, device=self._device)
            stretch_3 = one / stretch_factors
            return torch.concat((stretch_1, stretch_2, stretch_3), dim=1)

        stretches = []
        if not torch.numel(stretch_factors_ut):
            stretches += [calculate_stretches_ut(stretch_factors_ut)]
        if not torch.numel(stretch_factors_ebt):
            stretches += [calculate_stretches_ebt(stretch_factors_ebt)]
        if not torch.numel(stretch_factors_ps):
            stretches += [calculate_stretches_ps(stretch_factors_ps)]

        return torch.vstack(stretches)

    def _calculate_stress(
        self, stretches: Stretches, parameters: Parameters
    ) -> PiolaStress:
        deformation_gradient = self._assemble_deformation_gradient(stretches)
        strain_energy_gradient = grad(self._calculate_strain_energy, argnums=0)(
            deformation_gradient, parameters
        )
        dPsi_dF11 = strain_energy_gradient[0, 0]
        incompressibility_constraint = self._calculate_incompressibility_constraint(
            deformation_gradient, strain_energy_gradient
        )
        return dPsi_dF11 + incompressibility_constraint

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        cann_parameters, ogden_parameters = self._split_parameters(parameters)
        cann_strain_energy_terms = self._calculate_cann_strain_energy_terms(
            deformation_gradient, cann_parameters
        )
        ogden_strain_energy_terms = self._calculate_ogden_strain_energy_terms(
            deformation_gradient, ogden_parameters
        )
        return cann_strain_energy_terms + ogden_strain_energy_terms

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        start_index = 0
        stop_index = start_index + self._num_cann_parameters
        cann_parameters = parameters[start_index:stop_index]
        start_index = stop_index
        stop_index = start_index + self._num_ogden_parameters
        ogden_parameters = parameters[start_index:stop_index]
        return cann_parameters, ogden_parameters

    def _calculate_cann_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:

        def split_parameters(parameters: Parameters) -> SplittedParameters:
            return torch.chunk(parameters, self._num_cann_invariants)

        def calculate_strain_energy_terms(
            corrected_invariant: Invariant, parameters: Parameters
        ) -> StrainEnergy:

            def split_parameters(parameters: Parameters) -> SplittedParameters:
                return torch.chunk(parameters, self._num_cann_invariant_power_terms)

            def sum_invariant_power_terms(
                invariant_power: Invariant, parameters: Parameters
            ) -> StrainEnergy:
                one = torch.tensor(1.0, device=self._device)
                invariant = invariant_power

                param_1 = parameters[0]
                param_2 = parameters[1]
                param_3 = parameters[2]
                param_4 = parameters[3]
                param_5 = parameters[4]

                sub_term_1 = param_3 * invariant
                sub_term_2 = param_4 * (
                    torch.exp(param_1 * self._cann_exponential_weight_scale * invariant)
                    - one
                )
                sub_term_3 = param_5 * torch.log(one - param_2 * invariant)
                return sub_term_1 + sub_term_2 - sub_term_3

            invariant_identity = corrected_invariant
            invariant_squared = corrected_invariant**2
            parameters_identity, parameters_squared = split_parameters(parameters)

            terms_identity = sum_invariant_power_terms(
                invariant_identity, parameters_identity
            )
            terms_squared = sum_invariant_power_terms(
                invariant_squared, parameters_squared
            )
            return terms_identity + terms_squared

        cI_1, cI_2 = self._calculate_corrected_invariants(deformation_gradient)
        parameters_cI_1, parameters_cI_2 = split_parameters(parameters)
        strain_energy_cI_1 = calculate_strain_energy_terms(cI_1, parameters_cI_1)
        strain_energy_cI_2 = calculate_strain_energy_terms(cI_2, parameters_cI_2)
        return strain_energy_cI_1 + strain_energy_cI_2

    def _calculate_corrected_invariants(
        self, deformation_gradient: DeformationGradient
    ) -> Invariants:
        # Deformation tensors
        F = deformation_gradient
        C = torch.matmul(F.transpose(0, 1), F)  # right Cauchy-Green deformation tensor
        # Constants
        half = torch.tensor(1 / 2, device=self._device)
        three = torch.tensor(3.0, device=self._device)

        # Isotropic invariants
        I_1 = torch.trace(C)
        I_2 = half * (I_1**2 - torch.tensordot(C, C))
        I_1_cor = I_1 - three
        I_2_cor = I_2 - three
        return I_1_cor, I_2_cor

    def _calculate_ogden_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        three = torch.tensor(3.0, device=self._device)
        stretches = torch.diag(deformation_gradient)

        def calculate_ogden_terms(exponent: Tensor) -> StrainEnergy:
            return torch.sum(stretches**exponent) - three

        terms = vmap(calculate_ogden_terms)(self._ogden_exponents)
        weighted_terms = parameters * terms
        return torch.sum(weighted_terms)

    def _calculate_incompressibility_constraint(
        self,
        deformation_gradient: DeformationGradient,
        strain_energy_gradients: StrainEnergyGradient,
    ) -> IncompressibilityConstraint:
        one = torch.tensor(1.0, device=self._device)
        F_33 = deformation_gradient[2, 2]
        dPsi_dF33 = strain_energy_gradients[2, 2]
        pressure = F_33 * dPsi_dF33
        return -pressure * (one / F_33)

    def _assemble_deformation_gradient(
        self, stretches: Stretches
    ) -> DeformationGradient:
        zero = torch.tensor([0.0], device=self._device)
        F_11, F_22, F_33 = self._extract_stretches(stretches)
        row_1 = torch.concat((self._unsqueeze_zero_dimension(F_11), zero, zero))
        row_2 = torch.concat((zero, self._unsqueeze_zero_dimension(F_22), zero))
        row_3 = torch.concat((zero, zero, self._unsqueeze_zero_dimension(F_33)))
        return torch.stack((row_1, row_2, row_3))

    def _extract_stretches(self, stretches: Stretches) -> StretchesTuple:
        F_11 = stretches[0]
        F_22 = stretches[1]
        F_33 = stretches[2]
        return F_11, F_22, F_33

    def _unsqueeze_zero_dimension(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, dim=0)
