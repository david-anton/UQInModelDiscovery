import torch

from bayesianmdisc.models.orthotropiccann import OrthotropicCANN

device = torch.device("cpu")


def get_expected_parameter_names() -> tuple[str, ...]:
    return (
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
        "W_1_10 (l1, I_4f, p1, exp)",
        "W_1_12 (l1, I_4f, p2, exp)",
        "W_2_9 (l2, I_4f, p1, I)",
        "W_2_10 (l2, I_4f, p1, exp)",
        "W_2_11 (l2, I_4f, p2, I)",
        "W_2_12 (l2, I_4f, p2, exp)",
        "W_1_14 (l1, I_4s, p1, exp)",
        "W_1_16 (l1, I_4s, p2, exp)",
        "W_2_13 (l2, I_4s, p1, I)",
        "W_2_14 (l2, I_4s, p1, exp)",
        "W_2_15 (l2, I_4s, p2, I)",
        "W_2_16 (l2, I_4s, p2, exp)",
        "W_1_18 (l1, I_4n, p1, exp)",
        "W_1_20 (l1, I_4n, p2, exp)",
        "W_2_17 (l2, I_4n, p1, I)",
        "W_2_18 (l2, I_4n, p1, exp)",
        "W_2_19 (l2, I_4n, p2, I)",
        "W_2_20 (l2, I_4n, p2, exp)",
        "W_1_22 (l1, I_8fs, p1, exp)",
        "W_1_24 (l1, I_8fs, p2, exp)",
        "W_2_21 (l2, I_8fs, p1, I)",
        "W_2_22 (l2, I_8fs, p1, exp)",
        "W_2_23 (l2, I_8fs, p2, I)",
        "W_2_24 (l2, I_8fs, p2, exp)",
        "W_1_26 (l1, I_8fn, p1, exp)",
        "W_1_28 (l1, I_8fn, p2, exp)",
        "W_2_25 (l2, I_8fn, p1, I)",
        "W_2_26 (l2, I_8fn, p1, exp)",
        "W_2_27 (l2, I_8fn, p2, I)",
        "W_2_28 (l2, I_8fn, p2, exp)",
        "W_1_30 (l1, I_8sn, p1, exp)",
        "W_1_32 (l1, I_8sn, p2, exp)",
        "W_2_29 (l2, I_8sn, p1, I)",
        "W_2_30 (l2, I_8sn, p1, exp)",
        "W_2_31 (l2, I_8sn, p2, I)",
        "W_2_32 (l2, I_8sn, p2, exp)",
    )


def get_expected_initial_parameter_couplings() -> list[tuple[str, str]]:
    return [
        ("W_2_2 (l2, I_1, p1, exp)", "W_1_2 (l1, I_1, p1, exp)"),
        ("W_2_4 (l2, I_1, p2, exp)", "W_1_4 (l1, I_1, p2, exp)"),
        ("W_2_6 (l2, I_2, p1, exp)", "W_1_6 (l1, I_2, p1, exp)"),
        ("W_2_8 (l2, I_2, p2, exp)", "W_1_8 (l1, I_2, p2, exp)"),
        ("W_2_10 (l2, I_4f, p1, exp)", "W_1_10 (l1, I_4f, p1, exp)"),
        ("W_2_12 (l2, I_4f, p2, exp)", "W_1_12 (l1, I_4f, p2, exp)"),
        ("W_2_14 (l2, I_4s, p1, exp)", "W_1_14 (l1, I_4s, p1, exp)"),
        ("W_2_16 (l2, I_4s, p2, exp)", "W_1_16 (l1, I_4s, p2, exp)"),
        ("W_2_18 (l2, I_4n, p1, exp)", "W_1_18 (l1, I_4n, p1, exp)"),
        ("W_2_20 (l2, I_4n, p2, exp)", "W_1_20 (l1, I_4n, p2, exp)"),
        ("W_2_22 (l2, I_8fs, p1, exp)", "W_1_22 (l1, I_8fs, p1, exp)"),
        ("W_2_24 (l2, I_8fs, p2, exp)", "W_1_24 (l1, I_8fs, p2, exp)"),
        ("W_2_26 (l2, I_8fn, p1, exp)", "W_1_26 (l1, I_8fn, p1, exp)"),
        ("W_2_28 (l2, I_8fn, p2, exp)", "W_1_28 (l1, I_8fn, p2, exp)"),
        ("W_2_30 (l2, I_8sn, p1, exp)", "W_1_30 (l1, I_8sn, p1, exp)"),
        ("W_2_32 (l2, I_8sn, p2, exp)", "W_1_32 (l1, I_8sn, p2, exp)"),
    ]


def get_expected_initial_linear_parameters() -> tuple[str, ...]:
    return (
        "W_2_2 (l2, I_1, p1, exp)",
        "W_2_4 (l2, I_1, p2, exp)",
        "W_2_6 (l2, I_2, p1, exp)",
        "W_2_8 (l2, I_2, p2, exp)",
        "W_2_10 (l2, I_4f, p1, exp)",
        "W_2_12 (l2, I_4f, p2, exp)",
        "W_2_14 (l2, I_4s, p1, exp)",
        "W_2_16 (l2, I_4s, p2, exp)",
        "W_2_18 (l2, I_4n, p1, exp)",
        "W_2_20 (l2, I_4n, p2, exp)",
        "W_2_22 (l2, I_8fs, p1, exp)",
        "W_2_24 (l2, I_8fs, p2, exp)",
        "W_2_26 (l2, I_8fn, p1, exp)",
        "W_2_28 (l2, I_8fn, p2, exp)",
        "W_2_30 (l2, I_8sn, p1, exp)",
        "W_2_32 (l2, I_8sn, p2, exp)",
    )


def test_get_parameter_names() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.parameter_names

    expected = get_expected_parameter_names()
    assert expected == actual


def test_parameter_coupling_dict() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.inital_parameter_couplings

    expected = get_expected_initial_parameter_couplings()
    assert expected == actual


def test_linear_parameter_names() -> None:
    sut = OrthotropicCANN(device)

    actual = sut.initial_linear_parameters

    expected = get_expected_initial_linear_parameters()
    assert expected == actual
