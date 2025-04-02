import torch

from bayesianmdisc.models.linkacann import LinkaCANN

device = torch.device("cpu")


def get_expected_parameter_names() -> tuple[str, ...]:
    return (
        "W_1_2 (l1, I_1, p1, exp)",
        "W_1_4 (l1, I_1, p2, exp)",
        "W_2_1 (l2, I_1, p1, i)",
        "W_2_2 (l2, I_1, p1, exp)",
        "W_2_3 (l2, I_1, p2, i)",
        "W_2_4 (l2, I_1, p2, exp)",
        "W_1_6 (l1, I_2, p1, exp)",
        "W_1_8 (l1, I_2, p2, exp)",
        "W_2_5 (l2, I_2, p1, i)",
        "W_2_6 (l2, I_2, p1, exp)",
        "W_2_7 (l2, I_2, p2, i)",
        "W_2_8 (l2, I_2, p2, exp)",
        "W_1_10 (l1, I_4f, p1, exp)",
        "W_1_12 (l1, I_4f, p2, exp)",
        "W_2_9 (l2, I_4f, p1, i)",
        "W_2_10 (l2, I_4f, p1, exp)",
        "W_2_11 (l2, I_4f, p2, i)",
        "W_2_12 (l2, I_4f, p2, exp)",
        "W_1_14 (l1, I_4n, p1, exp)",
        "W_1_16 (l1, I_4n, p2, exp)",
        "W_2_13 (l2, I_4n, p1, i)",
        "W_2_14 (l2, I_4n, p1, exp)",
        "W_2_15 (l2, I_4n, p2, i)",
        "W_2_16 (l2, I_4n, p2, exp)",
    )


def test_get_parameter_names() -> None:
    sut = LinkaCANN(device)

    actual = sut.get_parameter_names()

    expected = get_expected_parameter_names()
    assert expected == actual
