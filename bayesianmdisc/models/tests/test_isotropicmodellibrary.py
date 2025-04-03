import torch

from bayesianmdisc.models.isotropicmodellibrary import IsotropicModelLibrary

device = torch.device("cpu")


def get_expected_parameter_names() -> tuple[str, ...]:
    def get_expected_cann_parameter_names() -> tuple[str, ...]:
        return (
            "W_1_2 (l1, I_1, p1, exp)",
            "W_1_3 (l1, I_1, p1, ln)",
            "W_2_1 (l2, I_1, p1, i)",
            "W_2_2 (l2, I_1, p1, exp)",
            "W_2_3 (l2, I_1, p1, ln)",
            "W_1_5 (l1, I_1, p2, exp)",
            "W_1_6 (l1, I_1, p2, ln)",
            "W_2_4 (l2, I_1, p2, i)",
            "W_2_5 (l2, I_1, p2, exp)",
            "W_2_6 (l2, I_1, p2, ln)",
            "W_1_8 (l1, I_2, p1, exp)",
            "W_1_9 (l1, I_2, p1, ln)",
            "W_2_7 (l2, I_2, p1, i)",
            "W_2_8 (l2, I_2, p1, exp)",
            "W_2_9 (l2, I_2, p1, ln)",
            "W_1_11 (l1, I_2, p2, exp)",
            "W_1_12 (l1, I_2, p2, ln)",
            "W_2_10 (l2, I_2, p2, i)",
            "W_2_11 (l2, I_2, p2, exp)",
            "W_2_12 (l2, I_2, p2, ln)",
        )

    def get_expected_ogden_parameter_names() -> tuple[str, ...]:
        parameter_names = []
        num_terms = 20
        min_exponent = torch.tensor(-5.0)
        max_exponent = torch.tensor(5.0)
        exponents = torch.linspace(min_exponent, max_exponent, num_terms).tolist()
        for index, exponent in zip(range(1, num_terms + 1), exponents):
            parameter_names += [f"O_{index} (exponent: {round(exponent,2)})"]
        return tuple(parameter_names)

    cann_parameter_names = get_expected_cann_parameter_names()
    ogden_parameter_names = get_expected_ogden_parameter_names()
    return cann_parameter_names + ogden_parameter_names


def test_get_parameter_names() -> None:
    sut = IsotropicModelLibrary(device)

    actual = sut.get_parameter_names()

    expected = get_expected_parameter_names()
    assert expected == actual
