import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device
from bayesianmdisc.data import DeformationInputs, StressOutputs, TestCases
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_simple_shear,
)
from bayesianmdisc.models.base import (
    AllowedTestCases,
    CauchyStresses,
    DeformationGradient,
    FlattenedCauchyStresses,
    FlattenedDeformationGradient,
    Invariant,
    Invariants,
    ParameterIndices,
    ParameterNames,
    ParameterPopulationMatrix,
    Parameters,
    SplittedParameters,
    StrainEnergy,
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
    validate_test_cases,
)


class OrthotropicCANN:

    def __init__(self, device: Device):
        self._device = device
        self._num_invariants = 8
        self._num_invariant_power_terms = 2
        self._num_activation_functions = 2
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ss = test_case_identifier_simple_shear
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [9]
        self._fiber_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._sheet_direction_reference = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._normal_direction_reference = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._zero_principal_stress_index = 1
        self._initial_num_parameters = self._determine_number_of_parameters()
        self._initial_parameter_names = self._init_parameter_names()
        self.output_dim = 9
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
        """The deformation input is expected to be a tensor corresponding to
        the flattened deformation gradient (of shape [n, 9])."""

        if validate_args:
            self._validate_inputs(inputs, test_cases, parameters)

        parameters = self._preprocess_parameters(parameters)

        def vmap_func(inputs_: FlattenedDeformationGradient) -> FlattenedCauchyStresses:
            return self._calculate_stresses(inputs_, parameters)

        return vmap(vmap_func)(inputs)

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

    def _determine_allowed_test_cases(self) -> AllowedTestCases:
        return torch.tensor(
            [self._test_case_identifier_bt, self._test_case_identifier_ss],
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

        (
            I_1_cor,
            I_2_cor,
            I_4f_cor,
            I_4s_cor,
            I_4n_cor,
            I_8fs_cor,
            I_8fn_cor,
            I_8sn_cor,
        ) = self._calculate_invariants(deformation_gradient)

        (
            params_I_1_cor,
            params_I_2_cor,
            params_I_4f_cor,
            params_I_4s_cor,
            params_I_4n_cor,
            params_I_8fs_cor,
            params_I_8fn_cor,
            params_I_8sn_cor,
        ) = self._split_parameters(parameters)

        return (
            calculate_strain_energy_terms_for_invariant(I_1_cor, params_I_1_cor)
            + calculate_strain_energy_terms_for_invariant(I_2_cor, params_I_2_cor)
            + calculate_strain_energy_terms_for_invariant(I_4f_cor, params_I_4f_cor)
            + calculate_strain_energy_terms_for_invariant(I_4s_cor, params_I_4s_cor)
            + calculate_strain_energy_terms_for_invariant(I_4n_cor, params_I_4n_cor)
            + calculate_strain_energy_terms_for_invariant(I_8fs_cor, params_I_8fs_cor)
            + calculate_strain_energy_terms_for_invariant(I_8fn_cor, params_I_8fn_cor)
            + calculate_strain_energy_terms_for_invariant(I_8sn_cor, params_I_8sn_cor)
        )

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
        I_4f = torch.inner(f, f)
        I_4s = torch.inner(s, s)
        I_4n = torch.inner(n, n)
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
