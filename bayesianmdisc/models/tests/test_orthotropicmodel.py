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
    "w (1, 10)",
    "w (1, 12)",
    "c (2, 9)",
    "c (2, 10)",
    "c (2, 11)",
    "c (2, 12)",
    "w (1, 14)",
    "w (1, 16)",
    "c (2, 13)",
    "c (2, 14)",
    "c (2, 15)",
    "c (2, 16)",
    "w (1, 18)",
    "w (1, 20)",
    "c (2, 17)",
    "c (2, 18)",
    "c (2, 19)",
    "c (2, 20)",
    "w (1, 22)",
    "w (1, 24)",
    "c (2, 21)",
    "c (2, 22)",
    "c (2, 23)",
    "c (2, 24)",
    "w (1, 26)",
    "w (1, 28)",
    "c (2, 25)",
    "c (2, 26)",
    "c (2, 27)",
    "c (2, 28)",
    "w (1, 30)",
    "w (1, 32)",
    "c (2, 29)",
    "c (2, 30)",
    "c (2, 31)",
    "c (2, 32)",
)
parameter_names_reduced = (
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
    "w (1, 12)",
    "c (2, 11)",
    "c (2, 12)",
    "w (1, 16)",
    "c (2, 15)",
    "c (2, 16)",
    "w (1, 20)",
    "c (2, 19)",
    "c (2, 20)",
    "w (1, 24)",
    "c (2, 23)",
    "c (2, 24)",
    "w (1, 28)",
    "c (2, 27)",
    "c (2, 28)",
    "w (1, 32)",
    "c (2, 31)",
    "c (2, 32)",
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
    ("c (2, 2)", "w (1, 2)"),
    ("c (2, 4)", "w (1, 4)"),
    ("c (2, 6)", "w (1, 6)"),
    ("c (2, 8)", "w (1, 8)"),
    ("c (2, 10)", "w (1, 10)"),
    ("c (2, 12)", "w (1, 12)"),
    ("c (2, 14)", "w (1, 14)"),
    ("c (2, 16)", "w (1, 16)"),
    ("c (2, 18)", "w (1, 18)"),
    ("c (2, 20)", "w (1, 20)"),
    ("c (2, 22)", "w (1, 22)"),
    ("c (2, 24)", "w (1, 24)"),
    ("c (2, 26)", "w (1, 26)"),
    ("c (2, 28)", "w (1, 28)"),
    ("c (2, 30)", "w (1, 30)"),
    ("c (2, 32)", "w (1, 32)"),
]
parameter_couplings_reduced = [
    ("c (2, 2)", "w (1, 2)"),
    ("c (2, 4)", "w (1, 4)"),
    ("c (2, 6)", "w (1, 6)"),
    ("c (2, 8)", "w (1, 8)"),
    ("c (2, 12)", "w (1, 12)"),
    ("c (2, 16)", "w (1, 16)"),
    ("c (2, 20)", "w (1, 20)"),
    ("c (2, 24)", "w (1, 24)"),
    ("c (2, 28)", "w (1, 28)"),
    ("c (2, 32)", "w (1, 32)"),
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
