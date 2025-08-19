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

# parameter_names_library = (
#     "MR (0, 1)",
#     "MR (1, 0) (NH)",
#     "MR (0, 2)",
#     "MR (1, 1)",
#     "MR (2, 0)",
#     "MR (0, 3)",
#     "MR (1, 2)",
#     "MR (2, 1)",
#     "MR (3, 0)",
#     "O (-5)",
#     "O (-4)",
#     "O (-3)",
#     "O (-1)",
#     "O (1)",
#     "O (3)",
#     "O (4)",
#     "O (5)",
# )
# parameter_names_cann = (
#     "W_1_2 (l1, I_1, p1, exp)",
#     "W_1_4 (l1, I_1, p2, exp)",
#     "W_2_1 (l2, I_1, p1, I)",
#     "W_2_2 (l2, I_1, p1, exp)",
#     "W_2_3 (l2, I_1, p2, I)",
#     "W_2_4 (l2, I_1, p2, exp)",
#     "W_1_6 (l1, I_2, p1, exp)",
#     "W_1_8 (l1, I_2, p2, exp)",
#     "W_2_5 (l2, I_2, p1, I)",
#     "W_2_6 (l2, I_2, p1, exp)",
#     "W_2_7 (l2, I_2, p2, I)",
#     "W_2_8 (l2, I_2, p2, exp)",
# )

parameter_names_library = (
    "c (0, 1)",
    "c (1, 0)",
    "c (0, 2)",
    "c (1, 1)",
    "c (2, 0)",
    "c (0, 3)",
    "c (1, 2)",
    "c (2, 1)",
    "c (3, 0)",
    "c (-5)",
    "c (-4)",
    "c (-3)",
    "c (-1)",
    "c (1)",
    "c (3)",
    "c (4)",
    "c (5)",
)
parameter_names_cann = (
    "w (1, 2)",
    "w (1, 4)",
    "c (2, 1)",
    "c (2, 2)",
    "c (2, 3)",
    "c (2, 4)",
    "w (1, 6)",
    "w (1, 8)",
    "c (2, 5)",
    "c (2, 6)",
    "c (2, 7)",
    "c (2, 8)",
)

num_parameters_library = len(parameter_names_library)
num_parameters_cann = len(parameter_names_cann)

parameter_couplings_cann = [
    ("c (2, 2)", "w (1, 2)"),
    ("c (2, 4)", "w (1, 4)"),
    ("c (2, 6)", "w (1, 6)"),
    ("c (2, 8)", "w (1, 8)"),
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
