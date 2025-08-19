import pytest
import torch

from bayesianmdisc.models.orthotropicmodel import (
    OrthotropicCANN,
    ParameterCouplingTuples,
    ParameterNames,
)

device = torch.device("cpu")

# parameter_names = (
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
#     "W_1_10 (l1, I_4f, p1, exp)",
#     "W_1_12 (l1, I_4f, p2, exp)",
#     "W_2_9 (l2, I_4f, p1, I)",
#     "W_2_10 (l2, I_4f, p1, exp)",
#     "W_2_11 (l2, I_4f, p2, I)",
#     "W_2_12 (l2, I_4f, p2, exp)",
#     "W_1_14 (l1, I_4s, p1, exp)",
#     "W_1_16 (l1, I_4s, p2, exp)",
#     "W_2_13 (l2, I_4s, p1, I)",
#     "W_2_14 (l2, I_4s, p1, exp)",
#     "W_2_15 (l2, I_4s, p2, I)",
#     "W_2_16 (l2, I_4s, p2, exp)",
#     "W_1_18 (l1, I_4n, p1, exp)",
#     "W_1_20 (l1, I_4n, p2, exp)",
#     "W_2_17 (l2, I_4n, p1, I)",
#     "W_2_18 (l2, I_4n, p1, exp)",
#     "W_2_19 (l2, I_4n, p2, I)",
#     "W_2_20 (l2, I_4n, p2, exp)",
#     "W_1_22 (l1, I_8fs, p1, exp)",
#     "W_1_24 (l1, I_8fs, p2, exp)",
#     "W_2_21 (l2, I_8fs, p1, I)",
#     "W_2_22 (l2, I_8fs, p1, exp)",
#     "W_2_23 (l2, I_8fs, p2, I)",
#     "W_2_24 (l2, I_8fs, p2, exp)",
#     "W_1_26 (l1, I_8fn, p1, exp)",
#     "W_1_28 (l1, I_8fn, p2, exp)",
#     "W_2_25 (l2, I_8fn, p1, I)",
#     "W_2_26 (l2, I_8fn, p1, exp)",
#     "W_2_27 (l2, I_8fn, p2, I)",
#     "W_2_28 (l2, I_8fn, p2, exp)",
#     "W_1_30 (l1, I_8sn, p1, exp)",
#     "W_1_32 (l1, I_8sn, p2, exp)",
#     "W_2_29 (l2, I_8sn, p1, I)",
#     "W_2_30 (l2, I_8sn, p1, exp)",
#     "W_2_31 (l2, I_8sn, p2, I)",
#     "W_2_32 (l2, I_8sn, p2, exp)",
# )
# parameter_names_reduced = (
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
#     "W_1_12 (l1, I_4f, p2, exp)",
#     "W_2_11 (l2, I_4f, p2, I)",
#     "W_2_12 (l2, I_4f, p2, exp)",
#     "W_1_16 (l1, I_4s, p2, exp)",
#     "W_2_15 (l2, I_4s, p2, I)",
#     "W_2_16 (l2, I_4s, p2, exp)",
#     "W_1_20 (l1, I_4n, p2, exp)",
#     "W_2_19 (l2, I_4n, p2, I)",
#     "W_2_20 (l2, I_4n, p2, exp)",
#     "W_1_24 (l1, I_8fs, p2, exp)",
#     "W_2_23 (l2, I_8fs, p2, I)",
#     "W_2_24 (l2, I_8fs, p2, exp)",
#     "W_1_28 (l1, I_8fn, p2, exp)",
#     "W_2_27 (l2, I_8fn, p2, I)",
#     "W_2_28 (l2, I_8fn, p2, exp)",
#     "W_1_32 (l1, I_8sn, p2, exp)",
#     "W_2_31 (l2, I_8sn, p2, I)",
#     "W_2_32 (l2, I_8sn, p2, exp)",
# )

parameter_names = (
    "w_1_2",
    "w_1_4",
    "c_2_1",
    "c_2_2",
    "c_2_3",
    "c_2_4",
    "w_1_6",
    "w_1_8",
    "c_2_5",
    "c_2_6",
    "c_2_7",
    "c_2_8",
    "w_1_10",
    "w_1_12",
    "c_2_9",
    "c_2_10",
    "c_2_11",
    "c_2_12",
    "w_1_14",
    "w_1_16",
    "c_2_13",
    "c_2_14",
    "c_2_15",
    "c_2_16",
    "w_1_18",
    "w_1_20",
    "c_2_17",
    "c_2_18",
    "c_2_19",
    "c_2_20",
    "w_1_22",
    "w_1_24",
    "c_2_21",
    "c_2_22",
    "c_2_23",
    "c_2_24",
    "w_1_26",
    "w_1_28",
    "c_2_25",
    "c_2_26",
    "c_2_27",
    "c_2_28",
    "w_1_30",
    "w_1_32",
    "c_2_29",
    "c_2_30",
    "c_2_31",
    "c_2_32",
)
parameter_names_reduced = (
    "w_1_2",
    "w_1_4",
    "c_2_1",
    "c_2_2",
    "c_2_3",
    "c_2_4",
    "w_1_6",
    "w_1_8",
    "c_2_5",
    "c_2_6",
    "c_2_7",
    "c_2_8",
    "w_1_12",
    "c_2_11",
    "c_2_12",
    "w_1_16",
    "c_2_15",
    "c_2_16",
    "w_1_20",
    "c_2_19",
    "c_2_20",
    "w_1_24",
    "c_2_23",
    "c_2_24",
    "w_1_28",
    "c_2_27",
    "c_2_28",
    "w_1_32",
    "c_2_31",
    "c_2_32",
)

num_parameters = len(parameter_names)
num_parameters_reduced = len(parameter_names_reduced)

# parameter_couplings = [
#     ("W_2_2 (l2, I_1, p1, exp)", "W_1_2 (l1, I_1, p1, exp)"),
#     ("W_2_4 (l2, I_1, p2, exp)", "W_1_4 (l1, I_1, p2, exp)"),
#     ("W_2_6 (l2, I_2, p1, exp)", "W_1_6 (l1, I_2, p1, exp)"),
#     ("W_2_8 (l2, I_2, p2, exp)", "W_1_8 (l1, I_2, p2, exp)"),
#     ("W_2_10 (l2, I_4f, p1, exp)", "W_1_10 (l1, I_4f, p1, exp)"),
#     ("W_2_12 (l2, I_4f, p2, exp)", "W_1_12 (l1, I_4f, p2, exp)"),
#     ("W_2_14 (l2, I_4s, p1, exp)", "W_1_14 (l1, I_4s, p1, exp)"),
#     ("W_2_16 (l2, I_4s, p2, exp)", "W_1_16 (l1, I_4s, p2, exp)"),
#     ("W_2_18 (l2, I_4n, p1, exp)", "W_1_18 (l1, I_4n, p1, exp)"),
#     ("W_2_20 (l2, I_4n, p2, exp)", "W_1_20 (l1, I_4n, p2, exp)"),
#     ("W_2_22 (l2, I_8fs, p1, exp)", "W_1_22 (l1, I_8fs, p1, exp)"),
#     ("W_2_24 (l2, I_8fs, p2, exp)", "W_1_24 (l1, I_8fs, p2, exp)"),
#     ("W_2_26 (l2, I_8fn, p1, exp)", "W_1_26 (l1, I_8fn, p1, exp)"),
#     ("W_2_28 (l2, I_8fn, p2, exp)", "W_1_28 (l1, I_8fn, p2, exp)"),
#     ("W_2_30 (l2, I_8sn, p1, exp)", "W_1_30 (l1, I_8sn, p1, exp)"),
#     ("W_2_32 (l2, I_8sn, p2, exp)", "W_1_32 (l1, I_8sn, p2, exp)"),
# ]
# parameter_couplings_reduced = [
#     ("W_2_2 (l2, I_1, p1, exp)", "W_1_2 (l1, I_1, p1, exp)"),
#     ("W_2_4 (l2, I_1, p2, exp)", "W_1_4 (l1, I_1, p2, exp)"),
#     ("W_2_6 (l2, I_2, p1, exp)", "W_1_6 (l1, I_2, p1, exp)"),
#     ("W_2_8 (l2, I_2, p2, exp)", "W_1_8 (l1, I_2, p2, exp)"),
#     ("W_2_12 (l2, I_4f, p2, exp)", "W_1_12 (l1, I_4f, p2, exp)"),
#     ("W_2_16 (l2, I_4s, p2, exp)", "W_1_16 (l1, I_4s, p2, exp)"),
#     ("W_2_20 (l2, I_4n, p2, exp)", "W_1_20 (l1, I_4n, p2, exp)"),
#     ("W_2_24 (l2, I_8fs, p2, exp)", "W_1_24 (l1, I_8fs, p2, exp)"),
#     ("W_2_28 (l2, I_8fn, p2, exp)", "W_1_28 (l1, I_8fn, p2, exp)"),
#     ("W_2_32 (l2, I_8sn, p2, exp)", "W_1_32 (l1, I_8sn, p2, exp)"),
# ]

parameter_couplings = [
    ("c_2_2", "w_1_2"),
    ("c_2_4", "w_1_4"),
    ("c_2_6", "w_1_6"),
    ("c_2_8", "w_1_8"),
    ("c_2_10", "w_1_10"),
    ("c_2_12", "w_1_12"),
    ("c_2_14", "w_1_14"),
    ("c_2_16", "w_1_16"),
    ("c_2_18", "w_1_18"),
    ("c_2_20", "w_1_20"),
    ("c_2_22", "w_1_22"),
    ("c_2_24", "w_1_24"),
    ("c_2_26", "w_1_26"),
    ("c_2_28", "w_1_28"),
    ("c_2_30", "w_1_30"),
    ("c_2_32", "w_1_32"),
]
parameter_couplings_reduced = [
    ("c_2_2", "w_1_2"),
    ("c_2_4", "w_1_4"),
    ("c_2_6", "w_1_6"),
    ("c_2_8", "w_1_8"),
    ("c_2_12", "w_1_12"),
    ("c_2_16", "w_1_16"),
    ("c_2_20", "w_1_20"),
    ("c_2_24", "w_1_24"),
    ("c_2_28", "w_1_28"),
    ("c_2_32", "w_1_32"),
]


def _expected_parameter_names() -> list[tuple[bool, ParameterNames]]:
    return [
        (False, parameter_names),
        (True, parameter_names_reduced),
    ]


def _expected_num_parameters() -> list[tuple[bool, int]]:
    return [
        (False, num_parameters),
        (True, num_parameters_reduced),
    ]


def _expected_parameter_coupling_dict() -> list[tuple[bool, ParameterCouplingTuples]]:
    return [
        (False, parameter_couplings),
        (True, parameter_couplings_reduced),
    ]


@pytest.mark.parametrize(
    ("use_only_squared_anisotropic_invariants", "parameter_names"),
    _expected_parameter_names(),
)
def test_parameter_names(
    use_only_squared_anisotropic_invariants: bool, parameter_names: ParameterNames
) -> None:
    sut = OrthotropicCANN(device, use_only_squared_anisotropic_invariants)

    actual = sut.parameter_names

    expected = parameter_names
    assert expected == actual


@pytest.mark.parametrize(
    ("use_only_squared_anisotropic_invariants", "num_parameters"),
    _expected_num_parameters(),
)
def test_num_parameters(
    use_only_squared_anisotropic_invariants: bool, num_parameters: int
) -> None:
    sut = OrthotropicCANN(device, use_only_squared_anisotropic_invariants)

    actual = sut.num_parameters

    expected = num_parameters
    assert expected == actual


@pytest.mark.parametrize(
    ("use_only_squared_anisotropic_invariants", "parameter_couplings"),
    _expected_parameter_coupling_dict(),
)
def test_parameter_coupling_dict(
    use_only_squared_anisotropic_invariants: bool,
    parameter_couplings: ParameterCouplingTuples,
) -> None:
    sut = OrthotropicCANN(device, use_only_squared_anisotropic_invariants)

    actual = sut.parameter_couplings

    expected = parameter_couplings
    assert expected == actual
