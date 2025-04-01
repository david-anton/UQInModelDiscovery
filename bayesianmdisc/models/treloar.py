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
    StrainEnergyGradients,
    Stretch,
    Stretches,
    validate_deformation_input_dimension,
    validate_input_numbers,
    validate_parameters,
    validate_test_cases,
)

StretchInput: TypeAlias = Stretch | Stretches
StretchesTuple: TypeAlias = tuple[Stretch, Stretch, Stretch]


class TreloarCANN:

    def __init__(self, device: Device):
        self._device = device
        self._num_cann_invariants = 2
        self._num_cann_invariant_power_terms = 2
        self._num_cann_parameters_per_invariant_power_term = 5
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

        def vmap_func(inputs_: StretchInput, test_case_: TestCase) -> PiolaStresses:
            return self._calculate_stress(inputs_, test_case_, parameters)

        return vmap(vmap_func)(inputs, test_cases)

    def _determine_number_of_parameters(self) -> tuple[int, int]:
        num_cann_parameters = (
            self._num_cann_invariants
            * self._num_cann_invariant_power_terms
            * self._num_cann_parameters_per_invariant_power_term
        )
        num_ogden_parameters = self._num_ogden_terms
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

    def _calculate_stress(
        self, stretches: StretchInput, test_case: TestCase, parameters: Parameters
    ) -> PiolaStress:
        deformation_gradient = self._assemble_deformation_gradient(stretches, test_case)
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

                sub_term_1 = param_1 * invariant
                sub_term_2 = param_2 * (torch.exp(param_3 * invariant) - one)
                sub_term_3 = param_4 * torch.log(one - param_5 * invariant)
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
        self, stretches: StretchInput, test_case: TestCase
    ) -> DeformationGradient:
        zero = torch.tensor(0.0, device=self._device)

        F_11, F_22, F_33 = self._determine_stretches(stretches, test_case)
        row_1 = torch.concat((self._unsqueeze_zero_dimension(F_11), zero, zero))
        row_2 = torch.concat((zero, self._unsqueeze_zero_dimension(F_22), zero))
        row_3 = torch.concat((zero, zero, self._unsqueeze_zero_dimension(F_33)))
        return torch.stack((row_1, row_2, row_3))

    def _determine_stretches(
        self, stretches: StretchInput, test_case: TestCase
    ) -> StretchesTuple:
        num_stretches = len(stretches)
        if num_stretches == 3:
            F_11 = stretches[0]
            F_22 = stretches[1]
            F_33 = stretches[2]
        else:
            F_11, F_22, F_33 = self._calculate_stretches_from_factor(
                stretches, test_case
            )
        return F_11, F_22, F_33

    def _calculate_stretches_from_factor(
        self, stretch: Stretch, test_case: TestCase
    ) -> StretchesTuple:
        one = torch.tensor(1.0, device=self._device)
        stretch_factor = stretch
        if test_case == self._test_case_identifier_ut:
            F_11 = stretch_factor
            F_22 = F_33 = one / torch.sqrt(stretch_factor)
        elif test_case == self._test_case_identifier_ebt:
            F_11 = F_22 = stretch_factor
            F_33 = one / stretch_factor**2
        else:
            F_11 = stretch_factor
            F_22 = one
            F_33 = one / stretch_factor
        return F_11, F_22, F_33

    def _unsqueeze_zero_dimension(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, dim=0)
