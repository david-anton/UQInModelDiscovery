from typing import Protocol, TypeAlias

import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.errors import ModelError
from bayesianmdisc.models.base import (
    DeformationGradient,
    DeformationInputs,
    Invariant,
    Invariants,
    OutputSelectionMask,
    ParameterIndices,
    ParameterNames,
    ParameterPopulationMatrix,
    Parameters,
    ParameterScales,
    PiolaStress,
    PiolaStresses,
    SplittedParameters,
    StrainEnergy,
    StressOutputs,
    Stretch,
    Stretches,
    count_active_parameters,
    determine_initial_parameter_mask,
    filter_active_parameter_indices,
    filter_active_parameter_names,
    init_parameter_mask,
    init_parameter_population_matrix,
    map_parameter_names_to_indices,
    mask_and_populate_parameters,
    mask_parameters,
    update_parameter_population_matrix,
    validate_deformation_input,
    validate_input_number,
    validate_model_state,
    validate_parameters,
    validate_stress_output_dimension,
    validate_test_cases,
)
from bayesianmdisc.models.base_mechanics import (
    assemble_stretches_from_factors,
    assemble_stretches_from_incompressibility_assumption,
    calculate_pressure_from_incompressibility_constraint,
)
from bayesianmdisc.models.base_outputselection import (
    count_number_of_selected_outputs,
    determine_full_output_size,
    validate_full_output_size,
)
from bayesianmdisc.models.utility import unsqueeze_if_necessary
from bayesianmdisc.testcases import (
    AllowedTestCases,
    TestCases,
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)

StretchesTuple: TypeAlias = tuple[Stretch, Stretch, Stretch]
OgdenExponents: TypeAlias = list[float]
MRExponents: TypeAlias = list[list[int]]
ParameterCouplingTuples: TypeAlias = list[tuple[str, str]]


class SEF(Protocol):
    num_parameters: int
    parameter_names: ParameterNames
    parameter_scales: ParameterScales

    def __call__(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        pass

    def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None: ...

    def activate_parameters(self, parameter_indices: ParameterIndices) -> None: ...

    def reset_parameter_deactivations(self) -> None: ...

    def get_active_parameter_indices(self) -> ParameterIndices: ...

    def get_active_parameter_names(self) -> ParameterNames: ...

    def get_number_of_active_parameters(self) -> int: ...

    def reduce_to_activated_parameters(self) -> None: ...

    def reduce_model_to_parameter_names(
        self, parameter_names: ParameterNames
    ) -> None: ...

    def deactivate_all_parameters(self) -> None: ...

    def get_state(self) -> ParameterPopulationMatrix: ...

    def init_state(
        self, parameter_population_matrix: ParameterPopulationMatrix
    ) -> None: ...


class LibrarySEF:
    def __init__(self, device: Device):
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
        self._scale_all_parameters = 1.0
        self.num_parameters = self._initial_num_parameters
        self.parameter_names = self._initial_parameter_names
        self.parameter_scales = self._init_parameter_scales()
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
        self._parameter_population_matrix = init_parameter_population_matrix(
            self.num_parameters, self._device
        )

    def __call__(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        parameters = self._preprocess_parameters(parameters)
        return self._calculate_strain_energy(deformation_gradient, parameters)

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

    def reduce_model_to_parameter_names(self, parameter_names: ParameterNames) -> None:
        active_parameter_indices = map_parameter_names_to_indices(
            parameter_names_of_interest=parameter_names,
            model_parameter_names=self.parameter_names,
        )
        self._deactivate_all_parameters()
        self.activate_parameters(active_parameter_indices)
        self.reduce_to_activated_parameters()

    def get_state(self) -> ParameterPopulationMatrix:
        return self._parameter_population_matrix

    def init_state(
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

    def _determine_mr_exponents(self) -> MRExponents:
        exponents = []
        for n in range(1, self._degree_mr_terms + 1):
            exponents_cI_1 = torch.linspace(
                start=0, end=n, steps=n + 1, dtype=torch.int64
            ).reshape((-1, 1))
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

    def _init_parameter_scales(self) -> ParameterScales:
        return torch.full(
            (self.num_parameters,), self._scale_all_parameters, device=self._device
        )

    def _preprocess_parameters(self, parameters: Parameters) -> Parameters:
        return mask_and_populate_parameters(
            parameters, self._parameter_mask, self._parameter_population_matrix
        )

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
        cI_1, cI_2 = calculate_corrected_invariants(deformation_gradient, self._device)

        terms = torch.concat(
            [
                unsqueeze_zero_dimension(cI_1 ** exponents[0] * cI_2 ** exponents[1])
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
        stretches = extract_stretches(deformation_gradient)

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
        _, I_2 = calculate_invariants(deformation_gradient, self._device)
        three = torch.tensor(3.0, device=self._device)
        ln_feature_parameter = parameters[0]
        return ln_feature_parameter * torch.log(I_2 / three)

    def _deactivate_all_parameters(self) -> None:
        parameter_indices = list(range(self.num_parameters))
        self.deactivate_parameters(parameter_indices)


class CANNSEF:
    def __init__(self, device: Device):
        self._device = device
        self._num_invariants = 2
        self._num_power_terms = 2
        self._num_activation_functions = 2
        self._initial_num_parameters_per_invariant = (
            self._init_number_of_parameters_per_invariant()
        )
        self._initial_num_parameters = self._init_number_of_parameters()
        self._initial_parameter_names = self._init_parameter_names()
        self._scale_linear_parameters = 1.0
        self._scale_parameters_in_exponent = 1e-4
        self.num_parameters = self._initial_num_parameters
        self.parameter_names = self._initial_parameter_names
        self.parameter_scales = self._init_parameter_scales()
        self.parameter_couplings = self._init_parameter_couplings()
        self._parameter_mask = init_parameter_mask(self.num_parameters, self._device)
        self._parameter_population_matrix = init_parameter_population_matrix(
            self.num_parameters, self._device
        )

    def __call__(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:
        parameters = self._preprocess_parameters(parameters)
        return self._calculate_strain_energy(deformation_gradient, parameters)

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

    def reduce_model_to_parameter_names(self, parameter_names: ParameterNames) -> None:
        active_parameter_indices = map_parameter_names_to_indices(
            parameter_names_of_interest=parameter_names,
            model_parameter_names=self.parameter_names,
        )
        self._deactivate_all_parameters()
        self.activate_parameters(active_parameter_indices)
        self.reduce_to_activated_parameters()

    def get_state(self) -> ParameterPopulationMatrix:
        return self._parameter_population_matrix

    def init_state(
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

    def _init_number_of_parameters_per_invariant(self) -> int:
        num_activations = self._num_activation_functions
        num_nonidentity_activtions = num_activations - 1
        num_params_first_layer = self._num_power_terms * num_nonidentity_activtions
        num_params_second_layer = self._num_power_terms * num_activations
        return num_params_first_layer + num_params_second_layer

    def _init_number_of_parameters(self) -> int:
        return self._num_invariants * self._initial_num_parameters_per_invariant

    def _init_parameter_names(self) -> ParameterNames:
        invariant_names = ["I_1", "I_2"]
        power_term_names = ["p1", "p2"]
        activation_names = ["I", "exp"]

        first_layer_index = 1
        second_layer_index = 1

        parameter_names = []
        for invariant in invariant_names:
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

        return tuple(parameter_names)

    def _init_parameter_scales(self) -> ParameterScales:
        parameter_scales: list[ParameterScales] = []
        for _ in range(self._num_invariants):
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

    def _init_parameter_couplings(self) -> ParameterCouplingTuples:
        parameter_names = self.parameter_names
        pointer_index = 0
        parameter_coupling_tuples: ParameterCouplingTuples = []
        step_size = self._initial_num_parameters_per_invariant
        for _ in range(self._num_invariants):
            linear_param_1 = parameter_names[pointer_index + 3]
            linear_param_2 = parameter_names[pointer_index + 5]
            nonlinear_param_1 = parameter_names[pointer_index + 0]
            nonlinear_param_2 = parameter_names[pointer_index + 1]
            parameter_coupling_tuples += [(linear_param_1, nonlinear_param_1)]
            parameter_coupling_tuples += [(linear_param_2, nonlinear_param_2)]
            pointer_index += step_size
        return parameter_coupling_tuples

    def _expand_parameter_indices_by_coupled_indices(
        self, parameter_indices: ParameterIndices
    ) -> ParameterIndices:
        expanded_parameter_indices: ParameterIndices = []

        for parameter_index in parameter_indices:
            is_parameter_coupled = False
            parameter_name = self.parameter_names[parameter_index]

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
                    parameter_index_0 = self.parameter_names.index(parameter_name_0)
                    parameter_index_1 = self.parameter_names.index(parameter_name_1)
                    expanded_parameter_indices += [parameter_index_0, parameter_index_1]
                    is_parameter_coupled = True
                parameter_coupling_index += 1

            if not is_parameter_coupled:
                expanded_parameter_indices += [parameter_index]

        return expanded_parameter_indices

    def _preprocess_parameters(self, parameters: Parameters) -> Parameters:
        return mask_and_populate_parameters(
            parameters, self._parameter_mask, self._parameter_population_matrix
        )

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:

        def calculate_strain_energy_term_for_one_invariant(
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

        invariants = calculate_corrected_invariants(deformation_gradient, self._device)
        params_invariants = self._split_parameters(parameters)

        strain_energy_terms: list[StrainEnergy] = []
        for invariant, params_invariant in zip(invariants, params_invariants):
            strain_energy_terms += [
                unsqueeze_zero_dimension(
                    calculate_strain_energy_term_for_one_invariant(
                        invariant, params_invariant
                    )
                )
            ]

        return torch.sum(torch.concat(strain_energy_terms))

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        return torch.chunk(parameters, self._num_invariants)

    def _deactivate_all_parameters(self) -> None:
        parameter_indices = list(range(self.num_parameters))
        self.deactivate_parameters(parameter_indices)


def calculate_corrected_invariants(
    deformation_gradient: DeformationGradient, device: Device
) -> Invariants:
    # Invariants
    I_1, I_2 = calculate_invariants(deformation_gradient, device)
    # Constants
    three = torch.tensor(3.0, device=device)
    # Corrected invariants
    I_1_cor = I_1 - three
    I_2_cor = I_2 - three
    return I_1_cor, I_2_cor


def calculate_invariants(
    deformation_gradient: DeformationGradient, device: Device
) -> Invariants:
    # Deformation tensors
    F = deformation_gradient
    C = torch.matmul(F.transpose(0, 1), F)  # right Cauchy-Green deformation tensor
    # Constants
    half = torch.tensor(1 / 2, device=device)
    # Isotropic invariants
    I_1 = torch.trace(C)
    I_2 = half * (I_1**2 - torch.tensordot(C, C))
    return I_1, I_2


def extract_stretches(deformation_gradient: DeformationGradient) -> Stretches:
    return torch.diag(deformation_gradient)


def unsqueeze_zero_dimension(tensor: Tensor) -> Tensor:
    return torch.unsqueeze(tensor, dim=0)


class IsotropicModel:

    def __init__(self, strain_energy_function: SEF, output_dim: int, device: Device):
        self._strain_energy_function = strain_energy_function
        self._device = device
        self._test_case_identifier_ut = test_case_identifier_uniaxial_tension
        self._test_case_identifier_ebt = test_case_identifier_equibiaxial_tension
        self._test_case_identifier_bt = test_case_identifier_biaxial_tension
        self._test_case_identifier_ps = test_case_identifier_pure_shear
        self._allowed_test_cases = self._determine_allowed_test_cases()
        self._allowed_input_dimensions = [1, 2, 3]
        self._allowed_output_dimensions = [1, 2]
        self._zero_principal_stress_index = 2
        validate_stress_output_dimension(output_dim, self._allowed_output_dimensions)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_parameters(self) -> int:
        return self._strain_energy_function.num_parameters

    @property
    def parameter_names(self) -> ParameterNames:
        return self._strain_energy_function.parameter_names

    @property
    def parameter_scales(self) -> ParameterScales:
        return self._strain_energy_function.parameter_scales

    def __call__(
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

        stretches = self._assemble_stretches_if_necessary(inputs, test_cases)

        def vmap_func(stretches_: Stretches) -> PiolaStresses:
            return self._calculate_stress(stretches_, parameters)

        return vmap(vmap_func)(stretches)

    def deactivate_parameters(self, parameter_indices: ParameterIndices) -> None:
        self._strain_energy_function.deactivate_parameters(parameter_indices)

    def activate_parameters(self, parameter_indices: ParameterIndices) -> None:
        self._strain_energy_function.activate_parameters(parameter_indices)

    def reset_parameter_deactivations(self) -> None:
        self._strain_energy_function.reset_parameter_deactivations()

    def get_active_parameter_indices(self) -> ParameterIndices:
        return self._strain_energy_function.get_active_parameter_indices()

    def get_active_parameter_names(self) -> ParameterNames:
        return self._strain_energy_function.get_active_parameter_names()

    def get_number_of_active_parameters(self) -> int:
        return self._strain_energy_function.get_number_of_active_parameters()

    def reduce_to_activated_parameters(self) -> None:
        self._strain_energy_function.reduce_to_activated_parameters()

    def reduce_model_to_parameter_names(self, parameter_names: ParameterNames) -> None:
        self._strain_energy_function.reduce_model_to_parameter_names(parameter_names)

    def get_model_state(self) -> ParameterPopulationMatrix:
        return self._strain_energy_function.get_state()

    def init_model_state(
        self, parameter_population_matrix: ParameterPopulationMatrix
    ) -> None:
        self._strain_energy_function.init_state(parameter_population_matrix)

    def set_output_dimension(self, output_dim: int) -> None:
        validate_stress_output_dimension(output_dim, self._allowed_output_dimensions)
        self._output_dim = output_dim

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

    def _validate_inputs(
        self, inputs: DeformationInputs, test_cases: TestCases, parameters: Parameters
    ) -> None:
        validate_input_number(inputs, test_cases)
        validate_deformation_input(inputs, self._allowed_input_dimensions)
        validate_test_cases(test_cases, self._allowed_test_cases)
        validate_parameters(parameters, self.num_parameters)

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
        dW_dF = grad(self._strain_energy_function, argnums=0)(F, parameters)
        p = calculate_pressure_from_incompressibility_constraint(
            F, dW_dF, self._zero_principal_stress_index
        )

        P = dW_dF - p * F_inverse_transpose

        P11 = P[0, 0]
        P22 = P[1, 1]
        if self._output_dim == 1:
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
        row_1 = torch.concat((unsqueeze_zero_dimension(F_11), zero, zero))
        row_2 = torch.concat((zero, unsqueeze_zero_dimension(F_22), zero))
        row_3 = torch.concat((zero, zero, unsqueeze_zero_dimension(F_33)))
        return torch.stack((row_1, row_2, row_3), dim=0)

    def _split_stretches(self, stretches: Stretches) -> StretchesTuple:
        F_11 = stretches[0]
        F_22 = stretches[1]
        F_33 = stretches[2]
        return F_11, F_22, F_33

    def _deactivate_all_parameters(self) -> None:
        self._strain_energy_function.deactivate_all_parameters()


def create_strain_energy_function(
    strain_energy_function_type: str, device: Device
) -> SEF:
    if strain_energy_function_type == "library":
        return LibrarySEF(device)
    elif strain_energy_function_type == "cann":
        return CANNSEF(device)
    else:
        raise ModelError(
            f"""There is no implementation for the requested 
            strain energy function type: {strain_energy_function_type}"""
        )


def create_isotropic_model(
    strain_energy_function_type: str, output_dim: int, device: Device
) -> IsotropicModel:
    strain_energy_function = create_strain_energy_function(
        strain_energy_function_type, device
    )
    return IsotropicModel(
        strain_energy_function=strain_energy_function,
        output_dim=output_dim,
        device=device,
    )


class OutputSelectorTreloar:

    def __init__(
        self, test_cases: TestCases, model: IsotropicModel, device: Device
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
        full_output_dim = self._expected_full_output_size[0]
        return torch.full((full_output_dim,), True, device=self._device)
