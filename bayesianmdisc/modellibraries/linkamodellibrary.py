from typing import Protocol, TypeAlias

import torch
from torch import vmap
from torch.func import grad

from bayesianmdisc.errors import ModelLibraryError
from bayesianmdisc.modellibraries.base import (
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
)
from bayesianmdisc.types import Device


class LinkaOrthotropicIncompressibleCANN:

    def __init__(self, device: Device):
        self.output_dim = 9
        self.num_parameters = 48
        self._num_deformation_inputs = 9
        self._device = device
        self._num_invariants = 8
        self._fiber_direction_ref = torch.tensor([1.0, 0.0, 0.0], device=device)
        self._sheet_direction_ref = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._normal_direction_ref = torch.tensor([1.0, 0.0, 0.0], device=device)

    def __call__(self, inputs: Inputs, parameters: Parameters) -> Outputs:
        return self.forward(inputs, parameters)

    def forward(self, inputs: Inputs, parameters: Parameters) -> Outputs:
        self._validate_parameters(parameters)
        deformation_gradients = inputs[:, : self._num_deformation_inputs]
        hydrostatic_pressures = inputs[:, -1].reshape((-1, 1))

        def vmap_func(
            deformation_gradient: DeformationGradient,
            hydrostatic_pressure: HydrostaticPressure,
        ) -> CauchyStressTensor:
            strain_energy_gradient = grad(self._calculate_strain_energy, argnums=0)(
                deformation_gradient, parameters
            )
            deformation_gradient_transposed = deformation_gradient.transpose(0, 1)
            pressure_tensor = hydrostatic_pressure * torch.eye(3, device=self._device)

            return (
                torch.matmul(strain_energy_gradient, deformation_gradient_transposed)
                - pressure_tensor
            )

        return vmap(vmap_func)(deformation_gradients, hydrostatic_pressures)

    def _validate_parameters(self, parameters: Parameters) -> None:
        parameter_size = parameters.size
        expected_size = torch.Size([self.num_parameters])
        if not parameter_size() == expected_size:
            raise ModelLibraryError(
                f"""Size of parameters is expected to be {expected_size}, 
                but is {parameter_size}"""
            )

    def _calculate_strain_energy(
        self, deformation_gradient: DeformationGradient, parameters: Parameters
    ) -> StrainEnergy:

        def calculate_invaraint_terms(
            invariant: Invariant, parameters: Parameters
        ) -> StrainEnergy:
            one = torch.tensor(1.0, device=self._device)
            sub_term_1 = parameters[0] * invariant
            sub_term_2 = parameters[1] * (torch.exp(parameters[2] * invariant) - one)
            sub_term_3 = parameters[3] * invariant**2
            sub_term_4 = parameters[4] * (torch.exp(parameters[5] * invariant**2) - one)
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
            parameters_I_1_cor,
            parameters_I_2_cor,
            parameters_I_4f_cor,
            parameters_I_4s_cor,
            parameters_I_4n_cor,
            parameters_I_8fs_cor,
            parameters_I_8fn_cor,
            parameters_I_8sn_cor,
        ) = self._split_parameters(parameters)

        return (
            calculate_invaraint_terms(I_1_cor, parameters_I_1_cor)
            + calculate_invaraint_terms(I_2_cor, parameters_I_2_cor)
            + calculate_invaraint_terms(I_4f_cor, parameters_I_4f_cor)
            + calculate_invaraint_terms(I_4s_cor, parameters_I_4s_cor)
            + calculate_invaraint_terms(I_4n_cor, parameters_I_4n_cor)
            + calculate_invaraint_terms(I_8fs_cor, parameters_I_8fs_cor)
            + calculate_invaraint_terms(I_8fn_cor, parameters_I_8fn_cor)
            + calculate_invaraint_terms(I_8sn_cor, parameters_I_8sn_cor)
        )

    def _calculate_invariants(
        self, deformation_gradient: DeformationGradient
    ) -> Invariants:
        # Deformation tensors
        F = deformation_gradient
        b = torch.matmul(F, F.transpose(0, 1))  # left Cauchy-Green deformation tensor
        # Direction tensors
        f = torch.matmul(F, self._fiber_direction_ref)
        s = torch.matmul(F, self._sheet_direction_ref)
        n = torch.matmul(F, self._normal_direction_ref)
        # Constants
        one = torch.tensor(1.0, device=self._device)
        three = torch.tensor(3.0, device=self._device)

        # Isotropic invariants
        I_1 = torch.trace(b)
        I_2 = 1 / 2 * (I_1**2 - torch.inner(b, b))
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
