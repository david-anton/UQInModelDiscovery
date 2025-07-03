from typing import TypeAlias

import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device
from bayesianmdisc.errors import OutputSelectorError
from bayesianmdisc.models.base import (
    AllowedTestCases,
    CauchyStresses,
    DeformationGradient,
    DeformationInputs,
    FlattenedCauchyStresses,
    FlattenedDeformationGradient,
    Invariant,
    Invariants,
    OutputSelectionMask,
    ParameterIndices,
    ParameterNames,
    ParameterPopulationMatrix,
    Parameters,
    ParameterScales,
    SplittedInvariants,
    SplittedParameters,
    StrainEnergy,
    StressOutputs,
    count_active_parameters,
    determine_initial_parameter_mask,
    filter_active_parameter_indices,
    filter_active_parameter_names,
    init_parameter_mask,
    init_parameter_population_matrix,
    map_parameter_names_to_indices,
    filter_active_parameter_scales,
    mask_and_populate_parameters,
    mask_parameters,
    update_parameter_population_matrix,
    validate_deformation_input,
    validate_input_number,
    validate_model_state,
    validate_parameters,
    validate_test_cases,
)
from bayesianmdisc.models.base_mechanics import (
    calculate_pressure_from_incompressibility_constraint,
)
from bayesianmdisc.models.base_outputselection import (
    count_number_of_selected_outputs,
    determine_full_output_size,
    validate_full_output_size,
)
from bayesianmdisc.testcases import (
    TestCases,
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
)
from bayesianmdisc.utility import flatten_outputs

ParameterCouplingTuples: TypeAlias = list[tuple[str, str]]


class OrthotropicCANN:

    def __init__(self, device: Device, use_only_squared_anisotropic_invariants=False):
        self._device = device
        self._use_reduced_model = use_only_squared_anisotropic_invariants
        self._num_invariants_isotropic = 2
        self._num_invariants_anisotropic = 6
        self._num_power_terms_isotropic = 2
        if self._use_reduced_model:
            self._num_power_terms_anisotropic = 1
        else:
            self._num_power_terms_anisotropic = 2
        self._num_activation_functions = 2
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ss_12 = test_case_identifier_simple_shear_12
        self._test_case_identifier_ss_21 = test_case_identifier_simple_shear_21
        self._test_case_identifier_ss_13 = test_case_identifier_simple_shear_13
        self._test_case_identifier_ss_31 = test_case_identifier_simple_shear_31
        self._test_case_identifier_ss_23 = test_case_identifier_simple_shear_23
        self._test_case_identifier_ss_32 = test_case_identifier_simple_shear_32
        self._allowed_test_cases = self._init_allowed_test_cases()
        self._allowed_input_dimensions = [9]
        self._fiber_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._sheet_direction_reference = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._normal_direction_reference = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._zero_principal_stress_index = 1
        self._irrelevant_stress_component = 4
        (
            self._initial_num_parameters_per_invariant_isotropic,
            self._initial_num_parameters_per_invariant_anisotropic,
        ) = self._init_number_of_parameters_per_invariant()
        (
            self._initial_num_parameters_isotropic,
            self._initial_num_parameters_anisotropic,
        ) = self._init_number_of_parameters()
        self._initial_num_parameters = (
            self._initial_num_parameters_isotropic
            + self._initial_num_parameters_anisotropic
        )
        (
            self._initial_parameter_names_isotropic,
            self._initial_parameter_names_anisotropic,
        ) = self._init_parameter_names()
        self._initial_parameter_names = (
            self._initial_parameter_names_isotropic
            + self._initial_parameter_names_anisotropic
        )
        self._output_dim = 8
        self._num_parameters = self._initial_num_parameters
        self._parameter_names = self._initial_parameter_names
        self._scale_linear_parameters = 1.0
        self._scale_parameters_in_exponent = 1e-4
        self._parameter_scales = self._init_parameter_scales()
        self._parameter_mask = init_parameter_mask(self._num_parameters, self._device)
        self._parameter_population_matrix = init_parameter_population_matrix(
            self._num_parameters, self._device
        )
        self.parameter_couplings = self._init_parameter_couplings()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    @property
    def parameter_names(self) -> ParameterNames:
        return self._parameter_names

    @property
    def parameter_scales(self) -> ParameterScales:
        return self._parameter_scales

    def __call__(
        self,
        inputs: DeformationInputs,
        test_cases: TestCases,
        parameters: Parameters,
        validate_args=True,
    ) -> StressOutputs:
        """The deformation input is expected to be a tensor corresponding to
        the flattened deformation gradient (of shape [n, 9])."""

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        parameters = self._preprocess_parameters(parameters)

        def vmap_func(inputs_: FlattenedDeformationGradient) -> FlattenedCauchyStresses:
            return self._calculate_stresses(inputs_, parameters)

        flattened_stresses = vmap(vmap_func)(inputs)
        reduced_flattened_stresses = self._reduce_to_relevant_stresses(
            flattened_stresses
        )

        return reduced_flattened_stresses

    def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None:
        expanded_parameter_indices = self._expand_parameter_indices_by_coupled_indices(
            parameter_indices
        )
        mask_parameters(expanded_parameter_indices, self._parameter_mask, False)

    def activate_parameters(self, parameter_indices: ParameterIndices) -> None:
        expanded_parameter_indices = self._expand_parameter_indices_by_coupled_indices(
            parameter_indices
        )
        mask_parameters(expanded_parameter_indices, self._parameter_mask, True)

    def reset_parameter_deactivations(self) -> None:
        self._parameter_mask = init_parameter_mask(self._num_parameters, self._device)

    def get_active_parameter_indices(self) -> ParameterIndices:
        return filter_active_parameter_indices(self._parameter_mask)

    def get_active_parameter_names(self) -> ParameterNames:
        return filter_active_parameter_names(
            self._parameter_mask, self._parameter_names
        )

    def get_number_of_active_parameters(self) -> int:
        return count_active_parameters(self._parameter_mask)

    def reduce_to_activated_parameters(self) -> None:
        old_parameter_mask = self._parameter_mask

        def reduce_num_parameters() -> None:
            self._num_parameters = self.get_number_of_active_parameters()

        def reduce_parameter_names() -> None:
            self._parameter_names = self.get_active_parameter_names()

        def reduce_parameter_scales() -> None:
            self._parameter_scales = filter_active_parameter_scales(
                self._parameter_mask, self._parameter_scales
            )

        def reduce_parameter_mask() -> None:
            self._parameter_mask = init_parameter_mask(
                self._num_parameters, self._device
            )

        def reduce_parameter_population_matrix() -> None:
            self._parameter_population_matrix = update_parameter_population_matrix(
                self._parameter_population_matrix, old_parameter_mask
            )

        reduce_num_parameters()
        reduce_parameter_names()
        reduce_parameter_scales()
        reduce_parameter_mask()
        reduce_parameter_population_matrix()

    def reduce_model_to_parameter_names(self, parameter_names: ParameterNames) -> None:
        active_parameter_indices = map_parameter_names_to_indices(
            parameter_names_of_interest=parameter_names,
            model_parameter_names=self._parameter_names,
        )
        self._deactivate_all_parameters()
        self.activate_parameters(active_parameter_indices)
        self.reduce_to_activated_parameters()

    def get_model_state(self) -> ParameterPopulationMatrix:
        return self._parameter_population_matrix

    def init_model_state(
        self, parameter_population_matrix: ParameterPopulationMatrix
    ) -> None:
        population_matrix = parameter_population_matrix
        validate_model_state(population_matrix, self._initial_num_parameters)
        initial_parameter_mask = determine_initial_parameter_mask(population_matrix)

        def init_reuced_models_num_parameters() -> None:
            self._num_parameters = population_matrix.shape[1]

        def init_reduced_models_parameter_names() -> None:
            self._parameter_names = filter_active_parameter_names(
                initial_parameter_mask, self._initial_parameter_names
            )

        def init_reduced_models_parameter_mask() -> None:
            self._parameter_mask = init_parameter_mask(
                self._num_parameters, self._device
            )

        def init_reduced_models_population_matrix() -> None:
            self._parameter_population_matrix = population_matrix

        init_reuced_models_num_parameters()
        init_reduced_models_parameter_names()
        init_reduced_models_parameter_mask()
        init_reduced_models_population_matrix()

    def _init_allowed_test_cases(self) -> AllowedTestCases:
        return torch.tensor(
            [
                self._test_case_identifier_bt,
                self._test_case_identifier_ss_12,
                self._test_case_identifier_ss_21,
                self._test_case_identifier_ss_13,
                self._test_case_identifier_ss_31,
                self._test_case_identifier_ss_23,
                self._test_case_identifier_ss_32,
            ],
            device=self._device,
        )

    def _init_number_of_parameters_per_invariant(self) -> tuple[int, int]:
        num_activations = self._num_activation_functions
        num_nonidentity_activtions = num_activations - 1

        def calculate_number_of_parameters_per_invariant(num_power_terms: int) -> int:
            num_params_first_layer = num_power_terms * num_nonidentity_activtions
            num_params_second_layer = num_power_terms * num_activations
            return num_params_first_layer + num_params_second_layer

        num_parameters_isotropic = calculate_number_of_parameters_per_invariant(
            self._num_power_terms_isotropic
        )
        num_parameters_anisotropic = calculate_number_of_parameters_per_invariant(
            self._num_power_terms_anisotropic,
        )
        return num_parameters_isotropic, num_parameters_anisotropic

    def _init_number_of_parameters(self) -> tuple[int, int]:

        def calculate_number_of_parameters(
            num_invariants: int, num_parameter_per_invariant: int
        ) -> int:
            return num_invariants * num_parameter_per_invariant

        num_parameters_isotropic = calculate_number_of_parameters(
            self._num_invariants_isotropic,
            self._initial_num_parameters_per_invariant_isotropic,
        )
        num_parameters_anisotropic = calculate_number_of_parameters(
            self._num_invariants_anisotropic,
            self._initial_num_parameters_per_invariant_anisotropic,
        )
        return num_parameters_isotropic, num_parameters_anisotropic

    def _init_parameter_names(self) -> tuple[ParameterNames, ParameterNames]:
        invariant_names_isotropic = [
            "I_1",
            "I_2",
        ]
        invariant_names_anisotropic = [
            "I_4f",
            "I_4s",
            "I_4n",
            "I_8fs",
            "I_8fn",
            "I_8sn",
        ]
        power_term_names_isotropic = ["p1", "p2"]
        first_layer_index_offset_isotropic = 0
        second_layer_index_offset_isotropic = 0
        if self._use_reduced_model:
            first_layer_index_offset_anisotropic = 1
            second_layer_index_offset_anisotropic = 2
            power_term_names_anisotropic = ["p2"]
        else:
            first_layer_index_offset_anisotropic = 0
            second_layer_index_offset_anisotropic = 0
            power_term_names_anisotropic = ["p1", "p2"]
        activation_names = ["I", "exp"]

        first_layer_index = 1
        second_layer_index = 1

        def init_parameter_names(
            invariant_names: list[str],
            power_term_names: list[str],
            first_layer_index: int,
            second_layer_index: int,
            first_layer_index_offset: int = 0,
            second_layer_index_offset: int = 0,
        ) -> tuple[ParameterNames, int, int]:
            parameter_names = []
            for invariant in invariant_names:
                first_layer_index += first_layer_index_offset
                second_layer_index += second_layer_index_offset
                # first layer
                for power in power_term_names:
                    for activation in activation_names[1:]:
                        parameter_names += [
                            f"W_1_{2 *first_layer_index} (l1, {invariant}, {power}, {activation})"
                        ]
                        first_layer_index += 1

                # second layer
                for power in power_term_names:
                    for activation in activation_names:
                        parameter_names += [
                            f"W_2_{second_layer_index} (l2, {invariant}, {power}, {activation})"
                        ]
                        second_layer_index += 1

            return tuple(parameter_names), first_layer_index, second_layer_index

        parameter_names_isotropic, first_layer_index, second_layer_index = (
            init_parameter_names(
                invariant_names_isotropic,
                power_term_names_isotropic,
                first_layer_index,
                second_layer_index,
                first_layer_index_offset_isotropic,
                second_layer_index_offset_isotropic,
            )
        )
        parameter_names_anisotropic, first_layer_index, second_layer_index = (
            init_parameter_names(
                invariant_names_anisotropic,
                power_term_names_anisotropic,
                first_layer_index,
                second_layer_index,
                first_layer_index_offset_anisotropic,
                second_layer_index_offset_anisotropic,
            )
        )
        return parameter_names_isotropic, parameter_names_anisotropic

    def _init_parameter_scales(self) -> ParameterScales:

        def init_parameter_scales_for_all_invariant_terms(
            num_invariants: int,
        ) -> ParameterScales:
            parameter_scales: list[ParameterScales] = []
            for _ in range(num_invariants):
                parameter_scales += [
                    torch.tensor(
                        [
                            self._scale_parameters_in_exponent,
                            self._scale_parameters_in_exponent,
                            self._scale_linear_parameters,
                            self._scale_linear_parameters,
                            self._scale_linear_parameters,
                            self._scale_linear_parameters,
                        ]
                    )
                ]
            return torch.concat(parameter_scales).to(self._device)

        def init_parameter_scales_for_squared_invariant_terms_only(
            num_invariants: int,
        ) -> ParameterScales:
            parameter_scales: list[ParameterScales] = []
            for _ in range(num_invariants):
                parameter_scales += [
                    torch.tensor(
                        [
                            self._scale_parameters_in_exponent,
                            self._scale_linear_parameters,
                            self._scale_linear_parameters,
                        ]
                    )
                ]
            return torch.concat(parameter_scales).to(self._device)

        parameter_scales_isotropic = init_parameter_scales_for_all_invariant_terms(
            self._num_invariants_isotropic,
        )
        if self._use_reduced_model:
            init_parameter_scales_func_anisotropic = (
                init_parameter_scales_for_squared_invariant_terms_only
            )
        else:
            init_parameter_scales_func_anisotropic = (
                init_parameter_scales_for_all_invariant_terms
            )
        parameter_scales_anisotropic = init_parameter_scales_func_anisotropic(
            self._num_invariants_anisotropic,
        )
        return torch.concat((parameter_scales_isotropic, parameter_scales_anisotropic))

    def _init_parameter_couplings(self) -> ParameterCouplingTuples:

        def init_parameter_couplings_for_all_invariant_terms(
            parameter_names: ParameterNames,
            num_invariants: int,
            initial_num_parameters_per_invariant: int,
        ) -> ParameterCouplingTuples:
            pointer_index = 0
            parameter_coupling_tuples: ParameterCouplingTuples = []
            step_size = initial_num_parameters_per_invariant
            for _ in range(num_invariants):
                linear_param_1 = parameter_names[pointer_index + 3]
                linear_param_2 = parameter_names[pointer_index + 5]
                nonlinear_param_1 = parameter_names[pointer_index + 0]
                nonlinear_param_2 = parameter_names[pointer_index + 1]
                parameter_coupling_tuples += [(linear_param_1, nonlinear_param_1)]
                parameter_coupling_tuples += [(linear_param_2, nonlinear_param_2)]
                pointer_index += step_size
            return parameter_coupling_tuples

        def init_parameter_couplings_for_squared_invariant_terms_only(
            parameter_names: ParameterNames,
            num_invariants: int,
            initial_num_parameters_per_invariant: int,
        ) -> ParameterCouplingTuples:
            pointer_index = 0
            parameter_coupling_tuples: ParameterCouplingTuples = []
            step_size = initial_num_parameters_per_invariant
            for _ in range(num_invariants):
                linear_param = parameter_names[pointer_index + 2]
                nonlinear_param = parameter_names[pointer_index + 0]
                parameter_coupling_tuples += [(linear_param, nonlinear_param)]
                pointer_index += step_size
            return parameter_coupling_tuples

        parameter_couplings_isotropic = (
            init_parameter_couplings_for_all_invariant_terms(
                self._initial_parameter_names_isotropic,
                self._num_invariants_isotropic,
                self._initial_num_parameters_per_invariant_isotropic,
            )
        )
        if self._use_reduced_model:
            init_parameter_coupling_func_anisotropic = (
                init_parameter_couplings_for_squared_invariant_terms_only
            )
        else:
            init_parameter_coupling_func_anisotropic = (
                init_parameter_couplings_for_all_invariant_terms
            )
        parameter_couplings_anisotropic = init_parameter_coupling_func_anisotropic(
            self._initial_parameter_names_anisotropic,
            self._num_invariants_anisotropic,
            self._initial_num_parameters_per_invariant_anisotropic,
        )
        return parameter_couplings_isotropic + parameter_couplings_anisotropic

    def _expand_parameter_indices_by_coupled_indices(
        self, parameter_indices: ParameterIndices
    ) -> ParameterIndices:
        expanded_parameter_indices: ParameterIndices = []

        for parameter_index in parameter_indices:
            is_parameter_coupled = False
            parameter_name = self._parameter_names[parameter_index]

            num_parameter_couplings = len(self.parameter_couplings)
            parameter_coupling_index = 0
            while (
                is_parameter_coupled == False
                and parameter_coupling_index < num_parameter_couplings
            ):
                coupling_tuple = self.parameter_couplings[parameter_coupling_index]
                if parameter_name in coupling_tuple:
                    parameter_name_0 = coupling_tuple[0]
                    parameter_name_1 = coupling_tuple[1]
                    parameter_index_0 = self._parameter_names.index(parameter_name_0)
                    parameter_index_1 = self._parameter_names.index(parameter_name_1)
                    expanded_parameter_indices += [parameter_index_0, parameter_index_1]
                    is_parameter_coupled = True
                parameter_coupling_index += 1

            if not is_parameter_coupled:
                expanded_parameter_indices += [parameter_index]

        return expanded_parameter_indices

    def _deactivate_all_parameters(self) -> None:
        parameter_indices = list(range(self._num_parameters))
        self.deactivate_parameters(parameter_indices)

    def _validate_inputs(
        self, inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
    ) -> None:
        validate_input_number(inputs, test_cases)
        validate_deformation_input(inputs, self._allowed_input_dimensions)
        validate_test_cases(test_cases, self._allowed_test_cases)
        validate_parameters(parameters, self._num_parameters)

    def _preprocess_parameters(self, parameters: Parameters) -> Parameters:
        return mask_and_populate_parameters(
            parameters, self._parameter_mask, self._parameter_population_matrix
        )

    def _calculate_stresses(
        self,
        flattened_deformation_gradient: FlattenedDeformationGradient,
        parameters: Parameters,
    ) -> FlattenedCauchyStresses:
        F = self._reshape_deformation_gradient(flattened_deformation_gradient)
        F_transpose = F.transpose(0, 1)
        dW_dF = grad(self._calculate_strain_energy, argnums=0)(F, parameters)
        p = calculate_pressure_from_incompressibility_constraint(
            F, dW_dF, self._zero_principal_stress_index
        )
        I = torch.eye(3, device=self._device)
        sigma = torch.matmul(dW_dF, F_transpose) - p * I
        return self._flatten_cauchy_stress_tensor(sigma)

    def _reshape_deformation_gradient(
        self, flattened_deformation_gradient: FlattenedDeformationGradient
    ) -> DeformationGradient:
        return flattened_deformation_gradient.reshape((3, 3))

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:

        def calculate_strain_energy_term_for_all_invariant_terms(
            invariant: Invariant, parameters: Parameters
        ) -> StrainEnergy:
            one = torch.tensor(1.0, device=self._device)
            param_0 = parameters[0]
            param_1 = parameters[1]
            param_2 = parameters[2]
            param_3 = parameters[3]
            param_4 = parameters[4]
            param_5 = parameters[5]
            sub_term_1 = param_2 * invariant
            sub_term_2 = param_3 * (torch.exp(param_0 * invariant) - one)
            sub_term_3 = param_4 * invariant**2
            sub_term_4 = param_5 * (torch.exp(param_1 * invariant**2) - one)
            return sub_term_1 + sub_term_2 + sub_term_3 + sub_term_4

        def calculate_strain_energy_term_for_squared_invariant_terms_only(
            invariant: Invariant, parameters: Parameters
        ) -> StrainEnergy:
            one = torch.tensor(1.0, device=self._device)
            param_0 = parameters[0]
            param_1 = parameters[1]
            param_2 = parameters[2]
            sub_term_1 = param_1 * invariant**2
            sub_term_2 = param_2 * (torch.exp(param_0 * invariant**2) - one)
            return sub_term_1 + sub_term_2

        invariants_isotropic, invariants_anisotropic = self._split_invariants(
            self._calculate_invariants(deformation_gradient)
        )
        params_isotropic, params_anisotropic = self._split_parameters(parameters)
        params_invariants_isotropic = self._split_isotropic_parameters(params_isotropic)
        params_invariants_anisotropic = self._split_anisotropic_parameters(
            params_anisotropic
        )

        strain_energy_terms: list[StrainEnergy] = []
        for invariant, params_invariant in zip(
            invariants_isotropic, params_invariants_isotropic
        ):
            strain_energy_terms += [
                torch.unsqueeze(
                    calculate_strain_energy_term_for_all_invariant_terms(
                        invariant, params_invariant
                    ),
                    dim=0,
                )
            ]

        if self._use_reduced_model:
            calculate_strain_energy_term_for_anisotropic_invariants = (
                calculate_strain_energy_term_for_squared_invariant_terms_only
            )
        else:
            calculate_strain_energy_term_for_anisotropic_invariants = (
                calculate_strain_energy_term_for_all_invariant_terms
            )
        for invariant, params_invariant in zip(
            invariants_anisotropic, params_invariants_anisotropic
        ):
            strain_energy_terms += [
                torch.unsqueeze(
                    calculate_strain_energy_term_for_anisotropic_invariants(
                        invariant, params_invariant
                    ),
                    dim=0,
                )
            ]

        return torch.sum(torch.concat(strain_energy_terms))

    def _calculate_invariants(
        self, deformation_gradient: DeformationGradient
    ) -> Invariants:
        # Deformation tensors
        F = deformation_gradient
        F_transpose = F.transpose(0, 1)
        b = torch.matmul(F, F_transpose)  # left Cauchy-Green deformation tensor
        # Direction tensors
        f = torch.matmul(F, self._fiber_direction_reference)
        s = torch.matmul(F, self._sheet_direction_reference)
        n = torch.matmul(F, self._normal_direction_reference)
        # Constants
        half = torch.tensor(0.5, device=self._device)
        one = torch.tensor(1.0, device=self._device)
        three = torch.tensor(3.0, device=self._device)

        # Isotropic invariants
        I_1 = torch.trace(b)
        I_2 = half * (I_1**2 - torch.tensordot(b, b))
        I_1_cor = I_1 - three
        I_2_cor = I_2 - three

        # Anisotropic invariants
        I_4f = torch.maximum(torch.inner(f, f), one)
        I_4s = torch.maximum(torch.inner(s, s), one)
        I_4n = torch.maximum(torch.inner(n, n), one)
        I_4f_cor = I_4f - one
        I_4s_cor = I_4s - one
        I_4n_cor = I_4n - one

        # Coupling invariants
        I_8fs = torch.inner(f, s)
        I_8fn = torch.inner(f, n)
        I_8sn = torch.inner(s, n)
        I_8fs_cor = I_8fs
        I_8fn_cor = I_8fn
        I_8sn_cor = I_8sn
        return (
            I_1_cor,
            I_2_cor,
            I_4f_cor,
            I_4s_cor,
            I_4n_cor,
            I_8fs_cor,
            I_8fn_cor,
            I_8sn_cor,
        )

    def _split_invariants(self, invariants: Invariants) -> SplittedInvariants:
        invariants_isotropic = invariants[: self._num_invariants_isotropic]
        invariants_anisotropic = invariants[self._num_invariants_isotropic :]
        return invariants_isotropic, invariants_anisotropic

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        parameters_isotropic = parameters[: self._initial_num_parameters_isotropic]
        parameters_anisotropic = parameters[self._initial_num_parameters_isotropic :]
        return parameters_isotropic, parameters_anisotropic

    def _split_isotropic_parameters(
        self, parameters_isotropic: Parameters
    ) -> SplittedParameters:
        return torch.chunk(parameters_isotropic, self._num_invariants_isotropic)

    def _split_anisotropic_parameters(
        self, parameters_anisotropic: Parameters
    ) -> SplittedParameters:
        return torch.chunk(parameters_anisotropic, self._num_invariants_anisotropic)

    def _flatten_cauchy_stress_tensor(self, stresses: CauchyStresses) -> CauchyStresses:
        return stresses.reshape((-1))

    def _reduce_to_relevant_stresses(
        self, flattened_stresses: CauchyStresses
    ) -> CauchyStresses:
        return torch.concat(
            (
                flattened_stresses[:, : self._irrelevant_stress_component],
                flattened_stresses[:, self._irrelevant_stress_component + 1 :],
            ),
            dim=1,
        )


class OutputSelectorLinka:

    def __init__(
        self, test_cases: TestCases, model: OrthotropicCANN, device: Device
    ) -> None:
        self._test_cases = test_cases
        self._num_outputs = len(self._test_cases)
        self._single_full_output_dim = model._output_dim
        self._device = device
        self._expected_full_output_size = determine_full_output_size(
            self._num_outputs, self._single_full_output_dim
        )
        self._selection_mask = self._determine_output_selction_mask()
        self.total_num_selected_outputs = count_number_of_selected_outputs(
            self._selection_mask
        )

    def __call__(self, full_outputs: StressOutputs) -> StressOutputs:
        validate_full_output_size(full_outputs, self._expected_full_output_size)
        return torch.masked_select(full_outputs, self._selection_mask)

    def _determine_output_selction_mask(self) -> OutputSelectionMask:
        selection_mask_list: list[OutputSelectionMask] = []

        def _reshape(mask: OutputSelectionMask) -> OutputSelectionMask:
            return mask.reshape((1, -1))

        for test_case in self._test_cases:
            selection_mask = torch.full(
                (self._single_full_output_dim,), False, device=self._device
            )
            if test_case == test_case_identifier_simple_shear_12:
                # selection_mask[1] = True
                selection_mask[3] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_21:
                selection_mask[1] = True
                # selection_mask[3] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_13:
                # selection_mask[2] = True
                selection_mask[5] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_31:
                selection_mask[2] = True
                # selection_mask[5] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_23:
                # selection_mask[4] = True
                selection_mask[6] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_32:
                selection_mask[4] = True
                # selection_mask[6] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_biaxial_tension:
                selection_mask[0] = False  # True
                selection_mask[7] = False  # True
                selection_mask_list += _reshape(selection_mask)
            else:
                raise OutputSelectorError(
                    f"""There ist no implementation for the requested test case: {test_case}"""
                )

        selection_mask = torch.concat(selection_mask_list, dim=0)
        return flatten_outputs(selection_mask)
