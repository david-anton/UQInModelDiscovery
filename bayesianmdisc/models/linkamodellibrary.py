from typing import Protocol, TypeAlias

import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.errors import ModelLibraryError
from bayesianmdisc.models.base import (
    CauchyStress,
    CauchyStresses,
    CauchyStressTensor,
    DeformationGradient,
    HydrostaticPressure,
    Inputs,
    Invariant,
    Invariants,
    Outputs,
    Parameters,
    SplittedParameters,
    StrainEnergy,
    StrainEnergyDerivative,
    StrainEnergyDerivatives,
    Stretch,
    Stretches,
)
from bayesianmdisc.types import Device, Tensor

# class LinkaOrthotropicIncompressibleCANN:

#     def __init__(self, device: Device):
#         self.output_dim = 9
#         self.num_parameters = 48
#         self._num_deformation_inputs = 9
#         self._device = device
#         self._num_invariants = 8
#         self._fiber_direction_ref = torch.tensor([1.0, 0.0, 0.0], device=device)
#         self._sheet_direction_ref = torch.tensor([0.0, 1.0, 0.0], device=device)
#         self._normal_direction_ref = torch.tensor([1.0, 0.0, 0.0], device=device)

#     def __call__(self, inputs: Inputs, parameters: Parameters) -> Outputs:
#         return self.forward(inputs, parameters)

#     def forward(self, inputs: Inputs, parameters: Parameters) -> Outputs:
#         self._validate_parameters(parameters)
#         deformation_gradients = inputs[:, : self._num_deformation_inputs]
#         hydrostatic_pressures = inputs[:, -1].reshape((-1, 1))

#         def vmap_func(
#             deformation_gradient: DeformationGradient,
#             hydrostatic_pressure: HydrostaticPressure,
#         ) -> CauchyStressTensor:
#             deformation_gradient = reshape_deformation_gradient(deformation_gradient)
#             strain_energy_gradient = grad(self._calculate_strain_energy, argnums=0)(
#                 deformation_gradient, parameters
#             )
#             deformation_gradient_transposed = deformation_gradient.transpose(0, 1)
#             pressure_tensor = hydrostatic_pressure * torch.eye(3, device=self._device)

#             return (
#                 torch.matmul(strain_energy_gradient, deformation_gradient_transposed)
#                 - pressure_tensor
#             )

#         return vmap(vmap_func)(deformation_gradients, hydrostatic_pressures)

#     def _validate_parameters(self, parameters: Parameters) -> None:
#         parameter_size = parameters.size
#         expected_size = torch.Size([self.num_parameters])
#         if not parameter_size() == expected_size:
#             raise ModelLibraryError(
#                 f"""Size of parameters is expected to be {expected_size},
#                 but is {parameter_size}"""
#             )

#     def _calculate_strain_energy(
#         self, deformation_gradient: DeformationGradient, parameters: Parameters
#     ) -> StrainEnergy:

#         def calculate_invaraint_terms(
#             invariant: Invariant, parameters: Parameters
#         ) -> StrainEnergy:
#             one = torch.tensor(1.0, device=self._device)
#             sub_term_1 = parameters[0] * invariant
#             sub_term_2 = parameters[1] * (torch.exp(parameters[2] * invariant) - one)
#             sub_term_3 = parameters[3] * invariant**2
#             sub_term_4 = parameters[4] * (torch.exp(parameters[5] * invariant**2) - one)
#             return sub_term_1 + sub_term_2 + sub_term_3 + sub_term_4

#         (
#             I_1_cor,
#             I_2_cor,
#             I_4f_cor,
#             I_4s_cor,
#             I_4n_cor,
#             I_8fs_cor,
#             I_8fn_cor,
#             I_8sn_cor,
#         ) = self._calculate_invariants(deformation_gradient)

#         (
#             parameters_I_1_cor,
#             parameters_I_2_cor,
#             parameters_I_4f_cor,
#             parameters_I_4s_cor,
#             parameters_I_4n_cor,
#             parameters_I_8fs_cor,
#             parameters_I_8fn_cor,
#             parameters_I_8sn_cor,
#         ) = self._split_parameters(parameters)

#         return (
#             calculate_invaraint_terms(I_1_cor, parameters_I_1_cor)
#             + calculate_invaraint_terms(I_2_cor, parameters_I_2_cor)
#             + calculate_invaraint_terms(I_4f_cor, parameters_I_4f_cor)
#             + calculate_invaraint_terms(I_4s_cor, parameters_I_4s_cor)
#             + calculate_invaraint_terms(I_4n_cor, parameters_I_4n_cor)
#             + calculate_invaraint_terms(I_8fs_cor, parameters_I_8fs_cor)
#             + calculate_invaraint_terms(I_8fn_cor, parameters_I_8fn_cor)
#             + calculate_invaraint_terms(I_8sn_cor, parameters_I_8sn_cor)
#         )

#     def _calculate_invariants(
#         self, deformation_gradient: DeformationGradient
#     ) -> Invariants:
#         # Deformation tensors
#         F = deformation_gradient
#         b = torch.matmul(F, F.transpose(0, 1))  # left Cauchy-Green deformation tensor
#         # Direction tensors
#         f = torch.matmul(F, self._fiber_direction_ref)
#         s = torch.matmul(F, self._sheet_direction_ref)
#         n = torch.matmul(F, self._normal_direction_ref)
#         # Constants
#         one = torch.tensor(1.0, device=self._device)
#         three = torch.tensor(3.0, device=self._device)

#         # Isotropic invariants
#         I_1 = torch.trace(b)
#         I_2 = 1 / 2 * (I_1**2 - torch.tensordot(b, b))
#         I_1_cor = I_1 - three
#         I_2_cor = I_2 - three

#         # Anisotropic invariants
#         I_4f = torch.inner(f, f)
#         I_4s = torch.inner(s, s)
#         I_4n = torch.inner(n, n)
#         I_4f_cor = I_4f - one
#         I_4s_cor = I_4s - one
#         I_4n_cor = I_4n - one

#         # Coupling invariants
#         I_8fs = torch.inner(f, s)
#         I_8fn = torch.inner(f, n)
#         I_8sn = torch.inner(s, n)
#         I_8fs_cor = I_8fs
#         I_8fn_cor = I_8fn
#         I_8sn_cor = I_8sn
#         return (
#             I_1_cor,
#             I_2_cor,
#             I_4f_cor,
#             I_4s_cor,
#             I_4n_cor,
#             I_8fs_cor,
#             I_8fn_cor,
#             I_8sn_cor,
#         )

#     def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
#         return torch.chunk(parameters, self._num_invariants)

# def reshape_deformation_gradient(deformation_gradient: Tensor) -> Tensor:
#     return deformation_gradient.reshape((3, 3))


class LinkaCANN:

    def __init__(self, device: Device):
        self.output_dim = 2
        self.num_parameters = 24
        self._num_deformation_inputs = 2
        self._device = device
        self._num_invariants = 4
        self._fiber_direction_ref = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._normal_direction_ref = torch.tensor([0.0, 0.0, 1.0], device=device)

    def __call__(self, inputs: Inputs, parameters: Parameters) -> CauchyStresses:
        return self.forward(inputs, parameters)

    def forward(self, inputs: Inputs, parameters: Parameters) -> CauchyStresses:
        self._validate_parameters(parameters)
        stretches = inputs

        def vmap_func(_stretches: Stretches) -> CauchyStresses:
            return self._calculate_stresses(_stretches, parameters)

        return vmap(vmap_func)(stretches)

    def _validate_parameters(self, parameters: Parameters) -> None:
        parameter_size = parameters.size
        expected_size = torch.Size([self.num_parameters])
        if not parameter_size() == expected_size:
            raise ModelLibraryError(
                f"""Size of parameters is expected to be {expected_size}, 
                but is {parameter_size}"""
            )

    def _calculate_stresses(
        self, stretches: Stretches, parameters: Parameters
    ) -> StrainEnergy:

        def calculate_fiber_stress(
            dPsi_dI_1: StrainEnergyDerivative,
            dPsi_dI_2: StrainEnergyDerivative,
            dPsi_dI_4f: StrainEnergyDerivative,
            stretches: Stretches,
        ) -> CauchyStress:
            stretch_fiber, stretch_normal = self._split_stretches(stretches)
            stretch_sheet = self._calculat_stretch_sheet(stretches)
            two = torch.tensor(2.0, device=self._device)
            squared_stretch_fiber = stretch_fiber**2
            squared_stretch_sheet = stretch_sheet**2
            squared_stretch_normal = stretch_normal**2
            return (
                two * dPsi_dI_1 * (squared_stretch_fiber - squared_stretch_sheet)
                + two
                * dPsi_dI_2
                * (squared_stretch_fiber - squared_stretch_sheet)
                * squared_stretch_normal
                + two * dPsi_dI_4f * squared_stretch_fiber
            )

        def calculate_normal_stress(
            dPsi_dI_1: StrainEnergyDerivative,
            dPsi_dI_2: StrainEnergyDerivative,
            dPsi_dI_4n: StrainEnergyDerivative,
            stretches: Stretches,
        ) -> CauchyStress:
            stretch_fiber, stretch_normal = self._split_stretches(stretches)
            stretch_sheet = self._calculat_stretch_sheet(stretches)
            two = torch.tensor(2.0, device=self._device)
            squared_stretch_fiber = stretch_fiber**2
            squared_stretch_sheet = stretch_sheet**2
            squared_stretch_normal = stretch_normal**2
            return (
                two * dPsi_dI_1 * (squared_stretch_normal - squared_stretch_sheet)
                + two
                * dPsi_dI_2
                * (squared_stretch_normal - squared_stretch_sheet)
                * squared_stretch_fiber
                + two * dPsi_dI_4n * squared_stretch_normal
            )

        dPsi_dI_1, dPsi_dI_2, dPsi_dI_4f, dPsi_dI_4n = (
            self._calculate_strain_energy_derivatives(stretches, parameters)
        )

        stress_fiber = calculate_fiber_stress(
            dPsi_dI_1, dPsi_dI_2, dPsi_dI_4f, stretches
        )
        stress_normal = calculate_normal_stress(
            dPsi_dI_1, dPsi_dI_2, dPsi_dI_4n, stretches
        )
        return self._concatenate_stresses(stress_fiber, stress_normal)

    def _calculate_strain_energy_derivatives(
        self, stretches: Stretches, parameters: Parameters
    ) -> StrainEnergyDerivatives:

        def calculate_strain_energy_derivative(
            corrected_invariant: Invariant, parameters: Parameters
        ) -> StrainEnergyDerivative:
            param_1 = parameters[0]
            param_2 = parameters[1]
            param_3 = parameters[2]
            param_4 = parameters[3]
            param_5 = parameters[4]
            param_6 = parameters[5]

            sub_term_1 = param_3
            sub_term_2 = param_1 * param_4 * torch.exp(param_1 * corrected_invariant)
            sub_term_3 = (
                torch.tensor(2.0, device=self._device)
                * corrected_invariant
                * (
                    param_5
                    + param_2 * param_6 * torch.exp(param_2 * corrected_invariant**2)
                )
            )
            return sub_term_1 + sub_term_2 + sub_term_3

        I_1_cor, I_2_cor, I_4f_cor, I_4n_cor = self._calculate_invariants(stretches)
        (
            parameters_I_1_cor,
            parameters_I_2_cor,
            parameters_I_4f_cor,
            parameters_I_4n_cor,
        ) = self._split_parameters(parameters)

        dPsi_dI_1 = calculate_strain_energy_derivative(I_1_cor, parameters_I_1_cor)
        dPsi_dI_2 = calculate_strain_energy_derivative(I_2_cor, parameters_I_2_cor)
        dPsi_dI_4f = calculate_strain_energy_derivative(I_4f_cor, parameters_I_4f_cor)
        dPsi_dI_4n = calculate_strain_energy_derivative(I_4n_cor, parameters_I_4n_cor)
        return dPsi_dI_1, dPsi_dI_2, dPsi_dI_4f, dPsi_dI_4n

    def _calculate_invariants(self, stretches: Stretches) -> Invariants:
        # Deformation tensors
        F = self._assemble_deformation_gradient(stretches)
        b = torch.matmul(F, F.transpose(0, 1))  # left Cauchy-Green deformation tensor
        # Direction tensors
        f = torch.matmul(F, self._fiber_direction_ref)
        n = torch.matmul(F, self._normal_direction_ref)
        # Constants
        one = torch.tensor(1.0, device=self._device)
        three = torch.tensor(3.0, device=self._device)

        # Isotropic invariants
        I_1 = torch.trace(b)
        I_2 = 1 / 2 * (I_1**2 - torch.tensordot(b, b))
        I_1_cor = I_1 - three
        I_2_cor = I_2 - three

        # Anisotropic invariants
        I_4f = torch.inner(f, f)
        I_4n = torch.inner(n, n)
        I_4f_cor = I_4f - one
        I_4n_cor = I_4n - one

        return I_1_cor, I_2_cor, I_4f_cor, I_4n_cor

    def _assemble_deformation_gradient(
        self, stretches: Stretches
    ) -> DeformationGradient:
        stretch_fiber, stretch_normal = self._split_stretches(stretches)
        stretch_sheet = self._calculat_stretch_sheet(stretches)
        row_1 = torch.concat(
            (
                torch.unsqueeze(stretch_fiber, dim=0),
                torch.tensor([0.0], device=self._device),
                torch.tensor([0.0], device=self._device),
            )
        )
        row_2 = torch.concat(
            (
                torch.tensor([0.0], device=self._device),
                torch.unsqueeze(stretch_sheet, dim=0),
                torch.tensor([0.0], device=self._device),
            )
        )
        row_3 = torch.concat(
            (
                torch.tensor([0.0], device=self._device),
                torch.tensor([0.0], device=self._device),
                torch.unsqueeze(stretch_normal, dim=0),
            )
        )
        return torch.stack((row_1, row_2, row_3))

    def _split_parameters(self, parameters: Parameters) -> SplittedParameters:
        return torch.chunk(parameters, self._num_invariants)

    def _calculat_stretch_sheet(self, stretches: Stretches) -> Stretch:
        stretch_fiber, stretch_normal = self._split_stretches(stretches)
        one = torch.tensor(1.0, device=self._device)
        return one / (stretch_fiber * stretch_normal)

    def _split_stretches(self, stretches: Stretches) -> tuple[Stretch, Stretch]:
        stretch_fiber = stretches[0]
        stretch_normal = stretches[1]
        return stretch_fiber, stretch_normal

    def _concatenate_stresses(
        self, stress_fiber: CauchyStress, stress_normal: CauchyStress
    ) -> CauchyStresses:
        return torch.concat(
            (
                torch.unsqueeze(stress_fiber, dim=0),
                torch.unsqueeze(stress_normal, dim=0),
            )
        )


# class LinkaCANNLikelihood:
#     def __init__(
#         self,
#         model: LinkaCANN,
#         material_parameters: BayesianMaterialParameters,
#         material_parameter_indices_func: MaterialParametersFunc,
#         data_likelihood_parameters: DataLikelihoodParameters,
#         pinn_funcs: PINNFuncs,
#         training_dataset: TrainingDataset,
#         calibration_dataset: ConcatenatedCalibrationData,
#         error_calculator: ErrorCalculatorProtocol,
#         error_distribution_creator: ErrorDistributionCreatorProtocol,
#         device: Device,
#     ) -> None:
#         self._ansatz = model
#         self._material_parameters = material_parameters
#         self._material_parameter_indices_func = material_parameter_indices_func
#         self._data_likelihood_parameters = data_likelihood_parameters
#         self._pinn_funcs = pinn_funcs
#         self._error_calculator = error_calculator
#         self._error_distribution_creator = error_distribution_creator
#         self._device = device
#         self._data_pde, self._data_heatflux = self._unpack_training_dataset(
#             training_dataset
#         )
#         self._data_calibration = calibration_dataset
#         self._num_flattened_y_pde = len(self._data_pde.x_coor)
#         self._num_flattened_y_heatflux = len(self._data_heatflux.y_true)
#         self._num_flattened_y_data = len(self._data_calibration.outputs)
#         (
#             self._flattened_y_pde_true,
#             self._flattened_y_heatflux_true,
#             self._flattened_y_data_true,
#         ) = self._assemble_flattened_true_outputs()
#         self._parameter_indices_pde, self._parameter_indices_heatflux_bc = (
#             self._assemble_parameter_indices()
#         )
#         self._num_likelihood_parameter_sets = 2
#         self._likelihood_data = self._initialize_data_likelihood()
#         self.num_material_params = self._material_parameters.num_parameters
#         self.num_ansatz_params = self._ansatz.network.num_parameters
#         self.likelihood_term_names = ["PDE", "HFBC", "Data"]

#     def prob(self, parameters: Tensor) -> Prob:
#         with torch.no_grad():
#             return self._prob(parameters)

#     def log_prob(self, parameters: Tensor) -> LogProb:
#         with torch.no_grad():
#             return self._log_prob(parameters)[0]

#     def log_prob_with_grad(self, parameters: Tensor) -> LogProb:
#         return self._log_prob(parameters)[0]

#     def log_prob_and_individual_log_probs_with_grad(
#         self, parameters: Tensor
#     ) -> LogProbAndIndividualLogProbs:
#         log_prob, individual_log_probs = self._log_prob(parameters)
#         return log_prob, concat_zero_dimensional_tensors(individual_log_probs)

#     def grad_log_prob(self, parameters: Tensor) -> GradLogProb:
#         return torch.autograd.grad(
#             self._log_prob(parameters)[0],
#             parameters,
#             retain_graph=True,
#             create_graph=False,
#         )[0]

#     def mse_loss_statistics(self, parameters: Tensor) -> MSELossStatistics:
#         mses_pde, mses_heatflux, mses_data = self._calculate_mse_losses(parameters)
#         mean_mse_pde = torch.mean(mses_pde, dim=0).detach().cpu().item()
#         std_mse_pde = torch.std(mses_pde, dim=0).detach().cpu().item()
#         mean_mse_heatflux = torch.mean(mses_heatflux, dim=0).detach().cpu().item()
#         std_mse_heatflux = torch.std(mses_heatflux, dim=0).detach().cpu().item()
#         mean_mse_data = torch.mean(mses_data, dim=0).detach().cpu().item()
#         std_mse_data = torch.std(mses_data, dim=0).detach().cpu().item()
#         return MSELossStatistics(
#             mean_pde=mean_mse_pde,
#             std_pde=std_mse_pde,
#             mean_heatflux_bc=mean_mse_heatflux,
#             std_heatflux_bc=std_mse_heatflux,
#             mean_data=mean_mse_data,
#             std_data=std_mse_data,
#         )

#     def _unpack_training_dataset(
#         self, training_dataset: TrainingDataset
#     ) -> tuple[TrainingData2DCollocation, TrainingData2DHeatFluxBC]:
#         training_dataloader = DataLoader(
#             dataset=training_dataset,
#             batch_size=len(training_dataset),
#             collate_fn=training_dataset.get_collate_func(),
#         )
#         training_data_iter = iter(training_dataloader)
#         data_pde, data_heatflux_bc = next(training_data_iter)
#         return data_pde, data_heatflux_bc

#     def _assemble_flattened_true_outputs(self) -> Outputs:
#         y_pde_true = torch.zeros((self._num_flattened_y_pde,), device=self._device)
#         y_heatflux_true = flatten_tensor(self._data_heatflux.y_true)
#         y_data_true = flatten_tensor(self._data_calibration.outputs)
#         return y_pde_true, y_heatflux_true, y_data_true

#     def _assemble_parameter_indices(self) -> tuple[Tensor, Tensor]:
#         parameter_indices_pde = torch.tensor(
#             self._material_parameter_indices_func(self._data_pde.x_coor),
#             dtype=torch.int,
#             device=self._device,
#         )
#         parameter_indices_heatflux_bc = torch.tensor(
#             self._material_parameter_indices_func(self._data_heatflux.x_coor),
#             dtype=torch.int,
#             device=self._device,
#         )
#         return parameter_indices_pde, parameter_indices_heatflux_bc

#     def _prob(self, parameters: Tensor) -> Prob:
#         return torch.exp(self._log_prob(parameters)[0])

#     @abstractmethod
#     def _log_prob(self, parameters: Tensor) -> LogProbAndIndividualLogProbsTuple:
#         raise NotImplementedError()

#     def _initialize_data_likelihood(self) -> Likelihood:
#         likelihood_parameters = torch.tensor(
#             [
#                 self._data_likelihood_parameters.mean,
#                 self._data_likelihood_parameters.std,
#             ],
#             device=self._device,
#         )
#         return self._error_distribution_creator.create(
#             likelihood_parameters, total_num_errors=self._num_flattened_y_data
#         )

#     def _initialize_likelihoods(self, likelihood_parameters: Tensor) -> Likelihoods:
#         parameters_pde, parameters_heatflux = self._split_likelihood_parameters(
#             likelihood_parameters
#         )
#         likelihood_pde = self._error_distribution_creator.create(
#             parameters_pde, total_num_errors=self._num_flattened_y_pde
#         )
#         likelihood_heatflux = self._error_distribution_creator.create(
#             parameters_heatflux, total_num_errors=self._num_flattened_y_heatflux
#         )
#         return likelihood_pde, likelihood_heatflux

#     def _split_likelihood_parameters(
#         self, likelihood_parameters: Tensor
#     ) -> LikelihoodParameterSets:
#         return torch.chunk(
#             likelihood_parameters, chunks=self._num_likelihood_parameter_sets
#         )

#     def _calculate_errors(
#         self, material_parameter_corrections: Tensor, ansatz_parameters: Tensor
#     ) -> Residuals:
#         flattened_y_pde, flattened_y_heatflux, flattened_y_data = (
#             self._calculate_flattened_ansatz_outputs(
#                 material_parameter_corrections, ansatz_parameters
#             )
#         )
#         errors_pde = self._error_calculator.calculate(
#             flattened_y_pde, self._flattened_y_pde_true
#         )
#         errors_heatflux = self._error_calculator.calculate(
#             flattened_y_heatflux, self._flattened_y_heatflux_true
#         )
#         errors_data = self._error_calculator.calculate(
#             flattened_y_data, self._flattened_y_data_true
#         )
#         return errors_pde, errors_heatflux, errors_data

#     def _calculate_flattened_ansatz_outputs(
#         self, material_parameter_corrections: Tensor, ansatz_parameters: Tensor
#     ) -> Outputs:
#         x_param = self._material_parameters(material_parameter_corrections)

#         def y_pde_func() -> Tensor:
#             x_coor = self._data_pde.x_coor
#             _x_param = x_param[self._parameter_indices_pde].reshape((-1, 1))
#             heat_source = self._data_pde.f
#             return self._pinn_funcs.pde(
#                 self._ansatz, x_coor, _x_param, heat_source, ansatz_parameters
#             )

#         def y_heatflux_func() -> Tensor:
#             x_coor = self._data_heatflux.x_coor
#             _x_param = x_param[self._parameter_indices_heatflux_bc].reshape((-1, 1))
#             normal = self._data_heatflux.normal
#             return self._pinn_funcs.heatflux_bc(
#                 self._ansatz, x_coor, _x_param, normal, ansatz_parameters
#             )

#         def y_data_func() -> Tensor:
#             x_coor = self._data_calibration.inputs
#             return self._ansatz(x_coor, ansatz_parameters)

#         y_pde = flatten_tensor(y_pde_func())
#         y_heatflux = flatten_tensor(y_heatflux_func())
#         y_data = flatten_tensor(y_data_func())
#         return y_pde, y_heatflux, y_data

#     @abstractmethod
#     def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, ...]:
#         raise NotImplementedError()

#     def _calculate_mse_losses(self, parameters: Tensor) -> MSETuple:

#         def vmap_mse_losses(parameters_sample: Tensor) -> MSETuple:
#             (
#                 material_parameter_corrections,
#                 ansatz_parameters,
#             ) = self._extract_parameters_for_mse(parameters_sample)
#             flattened_y_pde, flattened_y_heatflux, flattened_y_data = (
#                 self._calculate_flattened_ansatz_outputs(
#                     material_parameter_corrections, ansatz_parameters
#                 )
#             )

#             mse_pde = self._error_calculator.calculate_mse(
#                 flattened_y_pde, self._flattened_y_pde_true
#             )
#             mse_heatflux = self._error_calculator.calculate_mse(
#                 flattened_y_heatflux, self._flattened_y_heatflux_true
#             )
#             mse_data = self._error_calculator.calculate_mse(
#                 flattened_y_data, self._flattened_y_data_true
#             )
#             return mse_pde, mse_heatflux, mse_data

#         mses_pde, mses_heatflux, mses_data = vmap(vmap_mse_losses)(parameters)
#         return mses_pde, mses_heatflux, mses_data

#     @abstractmethod
#     def _extract_parameters_for_mse(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
#         raise NotImplementedError()
