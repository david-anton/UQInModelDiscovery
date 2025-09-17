import torch

from bayesianmdisc.customtypes import Device
from bayesianmdisc.models.base import (
    DeformationGradient,
    Pressure,
    StrainEnergyDerivatives,
    Stretches,
)
from bayesianmdisc.testcases import (
    TestCases,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)


def assemble_stretches_from_factors(
    stretch_factors: Stretches, test_cases: TestCases, device: Device
):
    indices_ut = test_cases == test_case_identifier_uniaxial_tension
    indices_ebt = test_cases == test_case_identifier_equibiaxial_tension
    indices_ps = test_cases == test_case_identifier_pure_shear

    stretch_factors_ut = stretch_factors[indices_ut]
    stretch_factors_ebt = stretch_factors[indices_ebt]
    stretch_factors_ps = stretch_factors[indices_ps]

    one = torch.tensor(1.0, device=device)

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
        stretch_2 = torch.ones_like(stretch_factors, device=device)
        stretch_3 = one / stretch_factors
        return torch.concat((stretch_1, stretch_2, stretch_3), dim=1)

    all_stretches = []
    if not torch.numel(stretch_factors_ut) == 0:
        all_stretches += [calculate_stretches_ut(stretch_factors_ut)]
    if not torch.numel(stretch_factors_ebt) == 0:
        all_stretches += [calculate_stretches_ebt(stretch_factors_ebt)]
    if not torch.numel(stretch_factors_ps) == 0:
        all_stretches += [calculate_stretches_ps(stretch_factors_ps)]

    return torch.vstack(all_stretches)


def assemble_stretches_from_incompressibility_assumption(
    stretches: Stretches, device: Device
) -> Stretches:
    one = torch.tensor(1.0, device=device)

    def calculate_stretches(stretches: Stretches) -> Stretches:
        stretch_1 = stretches[:, 0].reshape((-1, 1))
        stretch_2 = stretches[:, 1].reshape((-1, 1))
        stretch_3 = one / (stretch_1 * stretch_2)
        return torch.concat((stretch_1, stretch_2, stretch_3), dim=1)

    return calculate_stretches(stretches)


def calculate_pressure_from_incompressibility_constraint(
    deformation_gradient: DeformationGradient,
    strain_energy_derivatives: StrainEnergyDerivatives,
    zero_principal_stress_index: int,
) -> Pressure:
    # dW_dF = strain_energy_derivatives
    # F_transpose = deformation_gradient.transpose(0, 1)
    # matmul_result = torch.matmul(dW_dF, F_transpose)
    # index = zero_principal_stress_index
    # return matmul_result[index, index]
    index = zero_principal_stress_index
    dW_dF = strain_energy_derivatives
    F_inverse_transpose = deformation_gradient.inverse().transpose(0, 1)
    return dW_dF[index, index] / F_inverse_transpose[index, index]
