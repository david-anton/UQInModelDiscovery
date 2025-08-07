import pytest
import torch

from bayesianmdisc.models.isotropicmodel import (
    CANNSEF,
    ParameterNames,
    create_isotropic_model,
)

output_dim = 1
device = torch.device("cpu")

library_sef_type = "library"
cann_sef_type = "cann"

parameter_names_library = (
    "C_0_1 (MR)",
    "C_1_0 (NH)",
    "C_0_2",
    "C_1_1",
    "C_2_0",
    "C_0_3",
    "C_1_2",
    "C_2_1",
    "C_3_0",
    "Ogden (-5.0)",
    "Ogden (-4.0)",
    "Ogden (-3.0)",
    "Ogden (-1.0)",
    "Ogden (1.0)",
    "Ogden (3.0)",
    "Ogden (4.0)",
    "Ogden (5.0)",
    "ln_I2",
)
parameter_names_cann = (
    "W_1_2 (l1, I_1, p1, exp)",
    "W_1_4 (l1, I_1, p2, exp)",
    "W_2_1 (l2, I_1, p1, I)",
    "W_2_2 (l2, I_1, p1, exp)",
    "W_2_3 (l2, I_1, p2, I)",
    "W_2_4 (l2, I_1, p2, exp)",
    "W_1_6 (l1, I_2, p1, exp)",
    "W_1_8 (l1, I_2, p2, exp)",
    "W_2_5 (l2, I_2, p1, I)",
    "W_2_6 (l2, I_2, p1, exp)",
    "W_2_7 (l2, I_2, p2, I)",
    "W_2_8 (l2, I_2, p2, exp)",
)

num_parameters_library = len(parameter_names_library)
num_parameters_cann = len(parameter_names_cann)

parameter_couplings_cann = [
    ("W_2_2 (l2, I_1, p1, exp)", "W_1_2 (l1, I_1, p1, exp)"),
    ("W_2_4 (l2, I_1, p2, exp)", "W_1_4 (l1, I_1, p2, exp)"),
    ("W_2_6 (l2, I_2, p1, exp)", "W_1_6 (l1, I_2, p1, exp)"),
    ("W_2_8 (l2, I_2, p2, exp)", "W_1_8 (l1, I_2, p2, exp)"),
]


def _expected_parameter_names() -> list[tuple[str, ParameterNames]]:
    return [
        (library_sef_type, parameter_names_library),
        (cann_sef_type, parameter_names_cann),
    ]


def _expected_num_parameters() -> list[tuple[str, int]]:
    return [
        (library_sef_type, num_parameters_library),
        (cann_sef_type, num_parameters_cann),
    ]


@pytest.mark.parametrize(
    ("strain_energy_function_type", "parameter_names"),
    _expected_parameter_names(),
)
def test_parameter_names(
    strain_energy_function_type: str, parameter_names: ParameterNames
) -> None:
    sut = create_isotropic_model(
        strain_energy_function_type=strain_energy_function_type,
        output_dim=output_dim,
        device=device,
    )

    actual = sut.parameter_names

    expected = parameter_names
    assert expected == actual


@pytest.mark.parametrize(
    ("strain_energy_function_type", "num_parameters"),
    _expected_num_parameters(),
)
def test_num_parameters(strain_energy_function_type: str, num_parameters: int) -> None:
    sut = create_isotropic_model(
        strain_energy_function_type=strain_energy_function_type,
        output_dim=output_dim,
        device=device,
    )

    actual = sut.num_parameters

    expected = num_parameters
    assert expected == actual


def test_parameter_coupling_dict_for_library_sef() -> None:
    sut = CANNSEF(device=device)
    actual = sut.parameter_couplings

    expected = parameter_couplings_cann
    assert expected == actual
