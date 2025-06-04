from typing import TypeAlias

import numpy as np
import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device, Tensor, TensorSize
from bayesianmdisc.data import DeformationInputs, StressOutputs, TestCases
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
)
from bayesianmdisc.utility import flatten_outputs
from bayesianmdisc.errors import OutputSelectorError
from bayesianmdisc.models.base import (
    AllowedTestCases,
    CauchyStresses,
    DeformationGradient,
    FlattenedCauchyStresses,
    FlattenedDeformationGradient,
    Invariant,
    Invariants,
    LSDesignMatrix,
    LSTargets,
    OutputSelectionMask,
    ParameterIndices,
    ParameterNames,
    ParameterPopulationMatrix,
    Parameters,
    SplittedParameters,
    StrainEnergy,
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
    validate_test_cases,
)
from bayesianmdisc.models.base_mechanics import (
    calculate_pressure_from_incompressibility_constraint,
)
from bayesianmdisc.models.base_outputselection import (
    validate_full_output_size,
    count_number_of_selected_outputs,
    determine_full_output_size,
)

ParameterCouplingTuples: TypeAlias = list[tuple[str, str]]


class OrthotropicCANN:

    def __init__(self, device: Device):
        self._device = device
        self._num_invariants = 8
        self._num_invariant_power_terms = 2
        self._num_activation_functions = 2
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ss_12 = test_case_identifier_simple_shear_12
        self._test_case_identifier_ss_21 = test_case_identifier_simple_shear_21
        self._test_case_identifier_ss_13 = test_case_identifier_simple_shear_13
        self._test_case_identifier_ss_31 = test_case_identifier_simple_shear_31
        self._test_case_identifier_ss_23 = test_case_identifier_simple_shear_23
        self._test_case_identifier_ss_32 = test_case_identifier_simple_shear_32
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [9]
        self._fiber_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._sheet_direction_reference = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._normal_direction_reference = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._zero_principal_stress_index = 1
        self._irrelevant_stress_component = 4
        self._initial_num_parameters = self._determine_number_of_parameters()
        self._initial_num_parameters_per_invariant = (
            self._determine_number_of_parameters_per_invariant()
        )
        self._initial_parameter_names = self._init_parameter_names()
        self.output_dim = 8
        self.num_parameters = self._initial_num_parameters
        self.parameter_names = self._initial_parameter_names
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
        self._parameter_population_matrix = init_parameter_population_matrix(
            self.num_parameters, self._device
        )
        self.initial_linear_parameters, self.inital_parameter_couplings = (
            self._init_linear_parameters_and_couplings(self._initial_parameter_names)
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
            linear_parameter_indices = self._determine_linear_parameter_indices()

            for parameter_index in linear_parameter_indices:
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

    def _determine_allowed_test_cases(self) -> AllowedTestCases:
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

    def _determine_number_of_parameters(self) -> int:
        num_invariants = self._num_invariants
        num_power_terms = self._num_invariant_power_terms
        num_activation_functions = self._num_activation_functions

        num_parameters_first_layer = (
            num_invariants * num_power_terms * (num_activation_functions - 1)
        )
        num_parameters_second_layer = (
            num_invariants * num_power_terms * num_activation_functions
        )
        return num_parameters_first_layer + num_parameters_second_layer

    def _determine_number_of_parameters_per_invariant(self) -> int:
        return int(round(self._initial_num_parameters / self._num_invariants))

    def _init_parameter_names(self) -> ParameterNames:
        parameter_names = []
        first_layer_indices = 1
        second_layer_indices = 1
        invariant_names = [
            "I_1",
            "I_2",
            "I_4f",
            "I_4s",
            "I_4n",
            "I_8fs",
            "I_8fn",
            "I_8sn",
        ]
        power_names = ["p1", "p2"]
        activation_function_names = ["I", "exp"]

        for invariant in invariant_names:
            # first layer
            for power in power_names:
                for activation_function in activation_function_names[1:]:
                    parameter_names += [
                        f"W_1_{2 *first_layer_indices} (l1, {invariant}, {power}, {activation_function})"
                    ]
                    first_layer_indices += 1

            # second layer
            for power in power_names:
                for activation_function in activation_function_names:
                    parameter_names += [
                        f"W_2_{second_layer_indices} (l2, {invariant}, {power}, {activation_function})"
                    ]
                    second_layer_indices += 1

        return tuple(parameter_names)

    def _init_linear_parameters_and_couplings(
        self, initial_parameter_names: ParameterNames
    ) -> tuple[ParameterNames, ParameterCouplingTuples]:
        linear_parameters: list[str] = []
        parameter_coupling_tuples: ParameterCouplingTuples = []
        step_size = self._initial_num_parameters_per_invariant
        start_index = 0
        end_index = step_size

        for _ in range(self._num_invariants):
            parameter_names = initial_parameter_names[start_index:end_index]
            linear_param_1 = parameter_names[3]
            linear_param_2 = parameter_names[5]
            nonlinear_param_1 = parameter_names[0]
            nonlinear_param_2 = parameter_names[1]
            linear_parameters += [linear_param_1, linear_param_2]
            parameter_coupling_tuples += [(linear_param_1, nonlinear_param_1)]
            parameter_coupling_tuples += [(linear_param_2, nonlinear_param_2)]
            start_index = end_index
            end_index += step_size

        return tuple(linear_parameters), parameter_coupling_tuples

    def _expand_parameter_indices_by_coupled_indices(
        self, parameter_indices: ParameterIndices
    ) -> ParameterIndices:
        expanded_parameter_indices: ParameterIndices = []

        for parameter_index in parameter_indices:
            is_parameter_coupled = False
            parameter_name = self.parameter_names[parameter_index]

            for coupling_tuple in self.inital_parameter_couplings:
                if parameter_name in coupling_tuple:
                    parameter_name_0 = coupling_tuple[0]
                    parameter_name_1 = coupling_tuple[1]
                    parameter_index_0 = self.parameter_names.index(parameter_name_0)
                    parameter_index_1 = self.parameter_names.index(parameter_name_1)
                    expanded_parameter_indices += [parameter_index_0, parameter_index_1]
                    is_parameter_coupled = True

            if not is_parameter_coupled:
                expanded_parameter_indices += [parameter_index]

        return expanded_parameter_indices

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

        def calculate_strain_energy_terms_for_invariant(
            invariant: Invariant, parameters: Parameters
        ) -> StrainEnergy:
            one = torch.tensor(1.0, device=self._device)
            param_1 = parameters[0]
            param_2 = parameters[1]
            param_3 = parameters[2]
            param_4 = parameters[3]
            param_5 = parameters[4]
            param_6 = parameters[5]
            sub_term_1 = param_3 * invariant
            sub_term_2 = param_4 * (torch.exp(param_1 * invariant) - one)
            sub_term_3 = param_5 * invariant**2
            sub_term_4 = param_6 * (torch.exp(param_2 * invariant**2) - one)
            return sub_term_1 + sub_term_2 + sub_term_3 + sub_term_4

        invariants = self._calculate_invariants(deformation_gradient)
        params_invariants = self._split_parameters(parameters)

        strain_energy_terms: list[StrainEnergy] = []
        for invariant, params_invariant in zip(invariants, params_invariants):
            strain_energy_terms += [
                torch.unsqueeze(
                    calculate_strain_energy_terms_for_invariant(
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
        b = torch.matmul(F, F.transpose(0, 1))  # left Cauchy-Green deformation tensor
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

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        return torch.chunk(parameters, self._num_invariants)

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

    def _determine_linear_parameter_indices(self) -> ParameterIndices:
        linear_parameter_indices = []
        for linear_parameter in self.initial_linear_parameters:
            if linear_parameter in self.parameter_names:
                parameter_index = self.parameter_names.index(linear_parameter)
                linear_parameter_indices += [parameter_index]
        return linear_parameter_indices

    def _deactivate_all_parameters(self) -> None:
        parameter_indices = list(range(self.num_parameters))
        self.deactivate_parameters(parameter_indices)


class OutputSelectorLinka:

    def __init__(
        self, test_cases: TestCases, model: OrthotropicCANN, device: Device
    ) -> None:
        self._test_cases = test_cases
        self._num_outputs = len(self._test_cases)
        self._single_full_output_dim = model.output_dim
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
                selection_mask[1] = True
                selection_mask[3] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_21:
                selection_mask[1] = True
                selection_mask[3] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_13:
                selection_mask[2] = True
                selection_mask[5] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_31:
                selection_mask[2] = True
                selection_mask[5] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_23:
                selection_mask[4] = True
                selection_mask[6] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_simple_shear_32:
                selection_mask[4] = True
                selection_mask[6] = True
                selection_mask_list += _reshape(selection_mask)
            elif test_case == test_case_identifier_biaxial_tension:
                selection_mask[0] = True
                selection_mask[7] = True
                selection_mask_list += _reshape(selection_mask)
            else:
                raise OutputSelectorError(
                    f"""There ist no implementation for the requested test case: {test_case}"""
                )

        selection_mask = torch.concat(selection_mask_list, dim=0)
        return flatten_outputs(selection_mask)


# class OrthotropicCANN:

#     def __init__(self, device: Device):
#         self._device = device
#         self._num_invariants = 14
#         self._num_invariant_power_terms = 2
#         self._num_activation_functions = 2
#         self._test_case_identifier_bt = test_case_identifier_biaxial_tension
#         self._test_case_identifier_ss_12 = test_case_identifier_simple_shear_12
#         self._test_case_identifier_ss_21 = test_case_identifier_simple_shear_21
#         self._test_case_identifier_ss_13 = test_case_identifier_simple_shear_13
#         self._test_case_identifier_ss_31 = test_case_identifier_simple_shear_31
#         self._test_case_identifier_ss_23 = test_case_identifier_simple_shear_23
#         self._test_case_identifier_ss_32 = test_case_identifier_simple_shear_32
#         self._allowed_test_cases = self._determine_allowed_test_cases()
#         self._allowed_input_dimensions = [9]
#         self._fiber_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
#         self._sheet_direction_reference = torch.tensor([0.0, 1.0, 0.0], device=device)
#         self._normal_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
#         self._zero_principal_stress_index = 1
#         self._irrelevant_stress_component = 4
#         self._initial_num_parameters = self._determine_number_of_parameters()
#         self._initial_num_parameters_per_invariant = (
#             self._determine_number_of_parameters_per_invariant()
#         )
#         self._initial_parameter_names = self._init_parameter_names()
#         self.output_dim = 8
#         self.num_parameters = self._initial_num_parameters
#         self.parameter_names = self._initial_parameter_names
#         self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
#         self._parameter_population_matrix = init_parameter_population_matrix(
#             self.num_parameters, self._device
#         )
#         self.initial_linear_parameters, self.inital_parameter_couplings = (
#             self._init_linear_parameters_and_couplings(self._initial_parameter_names)
#         )

#     def __call__(
#         self,
#         inputs: DeformationInputs,
#         test_cases: TestCases,
#         parameters: Parameters,
#         validate_args=True,
#     ) -> StressOutputs:
#         return self.forward(inputs, test_cases, parameters, validate_args)

#     def forward(
#         self,
#         inputs: DeformationInputs,
#         test_cases: TestCases,
#         parameters: Parameters,
#         validate_args=True,
#     ) -> StressOutputs:
#         """The deformation input is expected to be a tensor corresponding to
#         the flattened deformation gradient (of shape [n, 9])."""

#         if validate_args:
#             self._validate_inputs(inputs, test_cases, parameters)

#         parameters = self._preprocess_parameters(parameters)

#         def vmap_func(inputs_: FlattenedDeformationGradient) -> FlattenedCauchyStresses:
#             return self._calculate_stresses(inputs_, parameters)

#         flattened_stresses = vmap(vmap_func)(inputs)
#         reduced_flattened_stresses = self._reduce_to_relevant_stresses(
#             flattened_stresses
#         )

#         return reduced_flattened_stresses

#     def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None:
#         expanded_parameter_indices = self._expand_parameter_indices_by_coupled_indices(
#             parameter_indices
#         )
#         mask_parameters(expanded_parameter_indices, self._parameter_mask, False)

#     def activate_parameters(self, parameter_indices: ParameterIndices) -> None:
#         expanded_parameter_indices = self._expand_parameter_indices_by_coupled_indices(
#             parameter_indices
#         )
#         mask_parameters(expanded_parameter_indices, self._parameter_mask, True)

#     def reset_parameter_deactivations(self) -> None:
#         self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)

#     def get_active_parameter_indices(self) -> ParameterIndices:
#         return filter_active_parameter_indices(self._parameter_mask)

#     def get_active_parameter_names(self) -> ParameterNames:
#         return filter_active_parameter_names(self._parameter_mask, self.parameter_names)

#     def get_number_of_active_parameters(self) -> int:
#         return count_active_parameters(self._parameter_mask)

#     def reduce_to_activated_parameters(self) -> None:
#         old_parameter_mask = self._parameter_mask

#         def reduce_num_parameters() -> None:
#             self.num_parameters = self.get_number_of_active_parameters()

#         def reduce_parameter_names() -> None:
#             self.parameter_names = self.get_active_parameter_names()

#         def reduce_parameter_mask() -> None:
#             self._parameter_mask = init_parameter_mask(
#                 self.num_parameters, self._device
#             )

#         def reduce_parameter_population_matrix() -> None:
#             self._parameter_population_matrix = update_parameter_population_matrix(
#                 self._parameter_population_matrix, old_parameter_mask
#             )

#         reduce_num_parameters()
#         reduce_parameter_names()
#         reduce_parameter_mask()
#         reduce_parameter_population_matrix()

#     def get_model_state(self) -> ParameterPopulationMatrix:
#         return self._parameter_population_matrix

#     def init_model_state(
#         self, parameter_population_matrix: ParameterPopulationMatrix
#     ) -> None:
#         population_matrix = parameter_population_matrix
#         validate_model_state(population_matrix, self._initial_num_parameters)
#         initial_parameter_mask = determine_initial_parameter_mask(population_matrix)

#         def init_reuced_models_num_parameters() -> None:
#             self.num_parameters = population_matrix.shape[1]

#         def init_reduced_models_parameter_names() -> None:
#             self.parameter_names = filter_active_parameter_names(
#                 initial_parameter_mask, self._initial_parameter_names
#             )

#         def init_reduced_models_parameter_mask() -> None:
#             self._parameter_mask = init_parameter_mask(
#                 self.num_parameters, self._device
#             )

#         def init_reduced_models_population_matrix() -> None:
#             self._parameter_population_matrix = population_matrix

#         init_reuced_models_num_parameters()
#         init_reduced_models_parameter_names()
#         init_reduced_models_parameter_mask()
#         init_reduced_models_population_matrix()

#     def assemble_linear_system_of_equations(
#         self,
#         inputs: DeformationInputs,
#         test_cases: TestCases,
#         outputs: StressOutputs,
#         validate_args=True,
#     ) -> tuple[LSDesignMatrix, LSTargets]:
#         parameters = torch.ones((self.num_parameters,), device=self._device)

#         if validate_args:
#             self._validate_inputs(inputs, test_cases, parameters)

#         def flatten_outputs(outputs: Tensor) -> Tensor:
#             if outputs.dim() == 1:
#                 return outputs
#             else:
#                 return torch.transpose(outputs, 1, 0).ravel()

#         def assemble_design_matrix(
#             inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
#         ) -> LSDesignMatrix:
#             covariates = []
#             linear_parameter_indices = self._determine_linear_parameter_indices()

#             for parameter_index in linear_parameter_indices:
#                 self._deactivate_all_parameters()
#                 self.activate_parameters([parameter_index])
#                 outputs = self.forward(inputs, test_cases, parameters)
#                 flattened_outputs = flatten_outputs(outputs)
#                 covariates += [flattened_outputs.cpu().detach().numpy()]

#             self.reset_parameter_deactivations()
#             return np.vstack(covariates)

#         def assemble_targets(outputs: StressOutputs) -> LSTargets:
#             flattened_outputs = flatten_outputs(outputs)
#             return flattened_outputs.cpu().detach().numpy()

#         design_matrix = assemble_design_matrix(inputs, test_cases, parameters)
#         targets = assemble_targets(outputs)
#         return design_matrix, targets

#     def _determine_allowed_test_cases(self) -> AllowedTestCases:
#         return torch.tensor(
#             [
#                 self._test_case_identifier_bt,
#                 self._test_case_identifier_ss_12,
#                 self._test_case_identifier_ss_21,
#                 self._test_case_identifier_ss_13,
#                 self._test_case_identifier_ss_31,
#                 self._test_case_identifier_ss_23,
#                 self._test_case_identifier_ss_32,
#             ],
#             device=self._device,
#         )

#     def _determine_number_of_parameters(self) -> int:
#         num_invariants = self._num_invariants
#         num_power_terms = self._num_invariant_power_terms
#         num_activation_functions = self._num_activation_functions

#         num_parameters_first_layer = (
#             num_invariants * num_power_terms * (num_activation_functions - 1)
#         )
#         num_parameters_second_layer = (
#             num_invariants * num_power_terms * num_activation_functions
#         )
#         return num_parameters_first_layer + num_parameters_second_layer

#     def _determine_number_of_parameters_per_invariant(self) -> int:
#         return int(round(self._initial_num_parameters / self._num_invariants))

#     def _init_parameter_names(self) -> ParameterNames:
#         parameter_names = []
#         first_layer_indices = 1
#         second_layer_indices = 1
#         invariant_names = [
#             "I_1",
#             "I_2",
#             "I_4f",
#             "I_4s",
#             "I_4n",
#             "I_5f",
#             "I_5s",
#             "I_5n",
#             "I_8fs",
#             "I_8fn",
#             "I_8sn",
#             "I_9fs",
#             "I_9fn",
#             "I_9sn",
#         ]
#         power_names = ["p1", "p2"]
#         activation_function_names = ["I", "exp"]

#         for invariant in invariant_names:
#             # first layer
#             for power in power_names:
#                 for activation_function in activation_function_names[1:]:
#                     parameter_names += [
#                         f"W_1_{2 *first_layer_indices} (l1, {invariant}, {power}, {activation_function})"
#                     ]
#                     first_layer_indices += 1

#             # second layer
#             for power in power_names:
#                 for activation_function in activation_function_names:
#                     parameter_names += [
#                         f"W_2_{second_layer_indices} (l2, {invariant}, {power}, {activation_function})"
#                     ]
#                     second_layer_indices += 1

#         return tuple(parameter_names)

#     def _init_linear_parameters_and_couplings(
#         self, initial_parameter_names: ParameterNames
#     ) -> tuple[ParameterNames, ParameterCouplingTuples]:
#         linear_parameters: list[str] = []
#         parameter_coupling_tuples: ParameterCouplingTuples = []
#         step_size = self._initial_num_parameters_per_invariant
#         start_index = 0
#         end_index = step_size

#         for _ in range(self._num_invariants):
#             parameter_names = initial_parameter_names[start_index:end_index]
#             linear_param_1 = parameter_names[3]
#             linear_param_2 = parameter_names[5]
#             nonlinear_param_1 = parameter_names[0]
#             nonlinear_param_2 = parameter_names[1]
#             linear_parameters += [linear_param_1, linear_param_2]
#             parameter_coupling_tuples += [(linear_param_1, nonlinear_param_1)]
#             parameter_coupling_tuples += [(linear_param_2, nonlinear_param_2)]
#             start_index = end_index
#             end_index += step_size

#         return tuple(linear_parameters), parameter_coupling_tuples

#     def _expand_parameter_indices_by_coupled_indices(
#         self, parameter_indices: ParameterIndices
#     ) -> ParameterIndices:
#         expanded_parameter_indices: ParameterIndices = []

#         for parameter_index in parameter_indices:
#             is_parameter_coupled = False
#             parameter_name = self.parameter_names[parameter_index]

#             for coupling_tuple in self.inital_parameter_couplings:
#                 if parameter_name in coupling_tuple:
#                     parameter_name_0 = coupling_tuple[0]
#                     parameter_name_1 = coupling_tuple[1]
#                     parameter_index_0 = self.parameter_names.index(parameter_name_0)
#                     parameter_index_1 = self.parameter_names.index(parameter_name_1)
#                     expanded_parameter_indices += [parameter_index_0, parameter_index_1]
#                     is_parameter_coupled = True

#             if not is_parameter_coupled:
#                 expanded_parameter_indices += [parameter_index]

#         return expanded_parameter_indices

#     def _validate_inputs(
#         self, inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
#     ) -> None:
#         validate_input_numbers(inputs, test_cases)
#         validate_deformation_input_dimension(inputs, self._allowed_input_dimensions)
#         validate_test_cases(test_cases, self._allowed_test_cases)
#         validate_parameters(parameters, self.num_parameters)

#     def _preprocess_parameters(self, parameters: Parameters) -> Parameters:
#         return mask_and_populate_parameters(
#             parameters, self._parameter_mask, self._parameter_population_matrix
#         )

#     def _calculate_stresses(
#         self,
#         flattened_deformation_gradient: FlattenedDeformationGradient,
#         parameters: Parameters,
#     ) -> FlattenedCauchyStresses:
#         F = self._reshape_deformation_gradient(flattened_deformation_gradient)
#         F_transpose = F.transpose(0, 1)
#         dW_dF = grad(self._calculate_strain_energy, argnums=0)(F, parameters)
#         p = calculate_pressure_from_incompressibility_constraint(
#             F, dW_dF, self._zero_principal_stress_index
#         )
#         I = torch.eye(3, device=self._device)

#         sigma = torch.matmul(dW_dF, F_transpose) - p * I
#         return self._flatten_cauchy_stress_tensor(sigma)

#     def _reshape_deformation_gradient(
#         self, flattened_deformation_gradient: FlattenedDeformationGradient
#     ) -> DeformationGradient:
#         return flattened_deformation_gradient.reshape((3, 3))

#     def _calculate_strain_energy(
#         self, deformation_gradient: DeformationGradient, parameters: Parameters
#     ) -> StrainEnergy:

#         def calculate_strain_energy_terms_for_invariant(
#             invariant: Invariant, parameters: Parameters
#         ) -> StrainEnergy:
#             one = torch.tensor(1.0, device=self._device)
#             param_1 = parameters[0]
#             param_2 = parameters[1]
#             param_3 = parameters[2]
#             param_4 = parameters[3]
#             param_5 = parameters[4]
#             param_6 = parameters[5]
#             sub_term_1 = param_3 * invariant
#             sub_term_2 = param_4 * (torch.exp(param_1 * invariant) - one)
#             sub_term_3 = param_5 * invariant**2
#             sub_term_4 = param_6 * (torch.exp(param_2 * invariant**2) - one)
#             return sub_term_1 + sub_term_2 + sub_term_3 + sub_term_4

#         invariants = self._calculate_invariants(deformation_gradient)
#         params_invariants = self._split_parameters(parameters)

#         strain_energy_terms: list[StrainEnergy] = []
#         for invariant, params_invariant in zip(invariants, params_invariants):
#             strain_energy_terms += [
#                 torch.unsqueeze(
#                     calculate_strain_energy_terms_for_invariant(
#                         invariant, params_invariant
#                     ),
#                     dim=0,
#                 )
#             ]

#         return torch.sum(torch.concat(strain_energy_terms))

#     def _calculate_invariants(
#         self, deformation_gradient: DeformationGradient
#     ) -> Invariants:
#         # Deformation tensors
#         F = deformation_gradient
#         C = torch.matmul(F.transpose(0, 1), F)  # right Cauchy-Green deformation tensor
#         b = torch.matmul(F, F.transpose(0, 1))  # left Cauchy-Green deformation tensor
#         C_squared = torch.matmul(C, C)
#         # Direction tensors
#         f_0 = self._fiber_direction_reference
#         s_0 = self._sheet_direction_reference
#         n_0 = self._normal_direction_reference
#         f = torch.matmul(F, f_0)
#         s = torch.matmul(F, s_0)
#         n = torch.matmul(F, n_0)
#         # Constants
#         half = torch.tensor(0.5, device=self._device)
#         one = torch.tensor(1.0, device=self._device)
#         three = torch.tensor(3.0, device=self._device)

#         # Isotropic invariants
#         I_1 = torch.trace(b)
#         I_2 = half * (I_1**2 - torch.tensordot(b, b))
#         I_1_cor = I_1 - three
#         I_2_cor = I_2 - three

#         # Anisotropic invariants
#         I_4f = torch.maximum(torch.inner(f, f), one)
#         I_4s = torch.maximum(torch.inner(s, s), one)
#         I_4n = torch.maximum(torch.inner(n, n), one)
#         I_4f_cor = I_4f - one
#         I_4s_cor = I_4s - one
#         I_4n_cor = I_4n - one

#         I_5f = torch.maximum(torch.inner(f_0, torch.matmul(C_squared, f_0)), one)
#         I_5s = torch.maximum(torch.inner(s_0, torch.matmul(C_squared, s_0)), one)
#         I_5n = torch.maximum(torch.inner(n_0, torch.matmul(C_squared, n_0)), one)
#         I_5f_cor = I_5f - one
#         I_5s_cor = I_5s - one
#         I_5n_cor = I_5n - one

#         # Coupling invariants
#         I_8fs = torch.inner(f, s)
#         I_8fn = torch.inner(f, n)
#         I_8sn = torch.inner(s, n)
#         I_8fs_cor = I_8fs
#         I_8fn_cor = I_8fn
#         I_8sn_cor = I_8sn

#         I_9fs = torch.inner(f_0, torch.matmul(C_squared, s_0))
#         I_9fn = torch.inner(f_0, torch.matmul(C_squared, n_0))
#         I_9sn = torch.inner(s_0, torch.matmul(C_squared, n_0))
#         I_9fs_cor = I_9fs
#         I_9fn_cor = I_9fn
#         I_9sn_cor = I_9sn

#         return (
#             I_1_cor,
#             I_2_cor,
#             I_4f_cor,
#             I_4s_cor,
#             I_4n_cor,
#             I_5f_cor,
#             I_5s_cor,
#             I_5n_cor,
#             I_8fs_cor,
#             I_8fn_cor,
#             I_8sn_cor,
#             I_9fs_cor,
#             I_9fn_cor,
#             I_9sn_cor,
#         )

#     def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
#         return torch.chunk(parameters, self._num_invariants)

#     def _flatten_cauchy_stress_tensor(self, stresses: CauchyStresses) -> CauchyStresses:
#         return stresses.reshape((-1))

#     def _reduce_to_relevant_stresses(
#         self, flattened_stresses: CauchyStresses
#     ) -> CauchyStresses:
#         return torch.concat(
#             (
#                 flattened_stresses[:, : self._irrelevant_stress_component],
#                 flattened_stresses[:, self._irrelevant_stress_component + 1 :],
#             ),
#             dim=1,
#         )

#     def _determine_linear_parameter_indices(self) -> ParameterIndices:
#         linear_parameter_indices = []
#         for linear_parameter in self.initial_linear_parameters:
#             if linear_parameter in self.parameter_names:
#                 parameter_index = self.parameter_names.index(linear_parameter)
#                 linear_parameter_indices += [parameter_index]
#         return linear_parameter_indices

#     def _deactivate_all_parameters(self) -> None:
#         parameter_indices = list(range(self.num_parameters))
#         self.deactivate_parameters(parameter_indices)
