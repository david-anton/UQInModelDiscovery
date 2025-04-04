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
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.models.base import (
    DeformationGradient,
    IncompressibilityConstraint,
    Invariants,
    ParameterNames,
    Parameters,
    PiolaStress,
    PiolaStresses,
    SplittedParameters,
    StrainEnergy,
    StrainEnergyDerivatives,
    Stretch,
    Stretches,
    validate_deformation_input_dimension,
    validate_input_numbers,
    validate_parameters,
    validate_test_cases,
)

StretchesTuple: TypeAlias = tuple[Stretch, Stretch, Stretch]
OgdenExponents: TypeAlias = list[float]
MRExponents: TypeAlias = list[list[int]]


class IsotropicModelLibrary:

    def __init__(self, device: Device):
        self._device = device
        self._num_ogden_terms = 33
        self._min_ogden_exponent = torch.tensor(-4.0, device=self._device)
        self._max_ogden_exponent = torch.tensor(4.0, device=self._device)
        self._ogden_exponents = self._determine_ogden_exponents()
        self._degree_mr_terms = 3
        self._mr_exponents = self._determine_mr_exponents()
        self._test_case_identifier_ut = test_case_identifier_uniaxial_tension
        self._test_case_identifier_ebt = test_case_identifier_equibiaxial_tension
        self._test_case_identifier_ps = test_case_identifier_pure_shear
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [1, 3]
        self._num_ogden_parameters, self._num_mr_parameters = (
            self._determine_number_of_parameters()
        )
        self.output_dim = 1
        self.num_parameters = self._num_ogden_parameters + self._num_mr_parameters

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
        (2) a tensor containing only the stretch in the first dimension
            which correpsonds to the stretch factor (of shape [n, 1]).
        In case (2), the stretches in the second and third dimensions are calculated
        from the stretch factor depending on the test case."""

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        stretches = self._assemble_stretches_if_necessary(inputs, test_cases)

        def vmap_func(stretches_: Stretches) -> PiolaStresses:
            return self._calculate_stress(stretches_, parameters)

        return vmap(vmap_func)(stretches)

    def get_parameter_names(self) -> ParameterNames:

        def compose_ogden_parameter_names() -> ParameterNames:
            parameter_names = []
            for index, exponent in zip(
                range(1, self._num_ogden_terms + 1), self._ogden_exponents
            ):
                parameter_names += [f"O_{index} (exponent: {round(exponent,2)})"]
            return tuple(parameter_names)

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

        ogden_parameter_names = compose_ogden_parameter_names()
        mr_parameter_names = compose_mr_parameter_names()
        return ogden_parameter_names + mr_parameter_names

    def _determine_number_of_parameters(self) -> tuple[int, int]:

        def determine_number_of_ogden_parameters() -> int:
            return self._num_ogden_terms

        def determine_number_of_mr_parameters() -> int:
            num_parameters = 0
            for n in range(1, self._degree_mr_terms + 1):
                num_parameters += n + 1
            return num_parameters

        num_ogden_parameters = determine_number_of_ogden_parameters()
        num_mr_parameters = determine_number_of_mr_parameters()
        return num_ogden_parameters, num_mr_parameters

    def _determine_ogden_exponents(self) -> OgdenExponents:
        return torch.linspace(
            start=self._min_ogden_exponent,
            end=self._max_ogden_exponent,
            steps=self._num_ogden_terms,
        ).tolist()

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
        stretch_dim = stretches.shape[1]
        if stretch_dim == 1:
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
        if not torch.numel(stretch_factors_ut) == 0:
            stretches += [calculate_stretches_ut(stretch_factors_ut)]
        if not torch.numel(stretch_factors_ebt) == 0:
            stretches += [calculate_stretches_ebt(stretch_factors_ebt)]
        if not torch.numel(stretch_factors_ps) == 0:
            stretches += [calculate_stretches_ps(stretch_factors_ps)]

        return torch.vstack(stretches)

    def _calculate_stress(
        self, stretches: Stretches, parameters: Parameters
    ) -> PiolaStress:
        deformation_gradient = self._assemble_deformation_gradient(stretches)
        strain_energy_derivatives = grad(self._calculate_strain_energy, argnums=0)(
            deformation_gradient, parameters
        )

        incompressibility_constraint = self._calculate_incompressibility_constraint(
            deformation_gradient, strain_energy_derivatives
        )
        first_piola_stress_tensor = (
            strain_energy_derivatives + incompressibility_constraint
        )
        return first_piola_stress_tensor[0, 0]

    def _assemble_deformation_gradient(
        self, stretches: Stretches
    ) -> DeformationGradient:
        zero = torch.tensor([0.0], device=self._device)
        F_11, F_22, F_33 = self._split_stretches(stretches)
        row_1 = torch.concat((self._unsqueeze_zero_dimension(F_11), zero, zero))
        row_2 = torch.concat((zero, self._unsqueeze_zero_dimension(F_22), zero))
        row_3 = torch.concat((zero, zero, self._unsqueeze_zero_dimension(F_33)))
        return torch.stack((row_1, row_2, row_3))

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        ogden_parameters, mr_parameters = self._split_parameters(parameters)
        ogden_strain_energy_terms = self._calculate_ogden_strain_energy_terms(
            deformation_gradient, ogden_parameters
        )
        mr_strain_energy_terms = self._calculate_mr_strain_energy_terms(
            deformation_gradient, mr_parameters
        )
        return ogden_strain_energy_terms + mr_strain_energy_terms

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        ogden_parameters = parameters[: self._num_ogden_parameters]
        mr_parameters = parameters[self._num_ogden_parameters :]
        return ogden_parameters, mr_parameters

    def _calculate_ogden_strain_energy_terms(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        three = torch.tensor(3.0, device=self._device)
        stretches = self._extract_stretches(deformation_gradient)

        terms = torch.concat(
            [
                torch.sum(stretches**exponent, dim=0, keepdim=True) - three
                for exponent in self._ogden_exponents
            ]
        )

        weighted_terms = parameters * terms
        return torch.sum(weighted_terms)

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

    def _split_stretches(self, stretches: Stretches) -> StretchesTuple:
        F_11 = stretches[0]
        F_22 = stretches[1]
        F_33 = stretches[2]
        return F_11, F_22, F_33

    def _extract_stretches(
        self, deformation_gradient: DeformationGradient
    ) -> Stretches:
        return torch.diag(deformation_gradient)

    def _calculate_incompressibility_constraint(
        self,
        deformation_gradient: DeformationGradient,
        strain_energy_derivatives: StrainEnergyDerivatives,
    ) -> IncompressibilityConstraint:
        F_33 = deformation_gradient[2, 2]
        dW_dF33 = strain_energy_derivatives[2, 2]
        pressure = dW_dF33 * F_33
        return -pressure * deformation_gradient.inverse().transpose(0, 1)

    def _unsqueeze_zero_dimension(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, dim=0)
