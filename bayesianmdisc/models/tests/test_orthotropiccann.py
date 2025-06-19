import torch

from bayesianmdisc.models.orthotropiccann import OrthotropicCANN

device = torch.device("cpu")

parameter_names = (
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
    "W_1_10 (l1, I_4f, p2, exp)",
    "W_2_9 (l2, I_4f, p2, I)",
    "W_2_10 (l2, I_4f, p2, exp)",
    "W_1_12 (l1, I_4s, p2, exp)",
    "W_2_11 (l2, I_4s, p2, I)",
    "W_2_12 (l2, I_4s, p2, exp)",
    "W_1_14 (l1, I_4n, p2, exp)",
    "W_2_13 (l2, I_4n, p2, I)",
    "W_2_14 (l2, I_4n, p2, exp)",
    "W_1_16 (l1, I_8fs, p2, exp)",
    "W_2_15 (l2, I_8fs, p2, I)",
    "W_2_16 (l2, I_8fs, p2, exp)",
    "W_1_18 (l1, I_8fn, p2, exp)",
    "W_2_17 (l2, I_8fn, p2, I)",
    "W_2_18 (l2, I_8fn, p2, exp)",
    "W_1_20 (l1, I_8sn, p2, exp)",
    "W_2_19 (l2, I_8sn, p2, I)",
    "W_2_20 (l2, I_8sn, p2, exp)",
)
num_parameters = len(parameter_names)
parameter_couplings = [
    ("W_2_2 (l2, I_1, p1, exp)", "W_1_2 (l1, I_1, p1, exp)"),
    ("W_2_4 (l2, I_1, p2, exp)", "W_1_4 (l1, I_1, p2, exp)"),
    ("W_2_6 (l2, I_2, p1, exp)", "W_1_6 (l1, I_2, p1, exp)"),
    ("W_2_8 (l2, I_2, p2, exp)", "W_1_8 (l1, I_2, p2, exp)"),
    ("W_2_10 (l2, I_4f, p2, exp)", "W_1_10 (l1, I_4f, p2, exp)"),
    ("W_2_12 (l2, I_4s, p2, exp)", "W_1_12 (l1, I_4s, p2, exp)"),
    ("W_2_14 (l2, I_4n, p2, exp)", "W_1_14 (l1, I_4n, p2, exp)"),
    ("W_2_16 (l2, I_8fs, p2, exp)", "W_1_16 (l1, I_8fs, p2, exp)"),
    ("W_2_18 (l2, I_8fn, p2, exp)", "W_1_18 (l1, I_8fn, p2, exp)"),
    ("W_2_20 (l2, I_8sn, p2, exp)", "W_1_20 (l1, I_8sn, p2, exp)"),
]


def test_parameter_names() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.parameter_names

    expected = parameter_names
    assert expected == actual


def test_num_parameters() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.num_parameters

    expected = num_parameters
    assert expected == actual


def test_parameter_coupling_dict() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.parameter_couplings

    expected = parameter_couplings
    assert expected == actual
