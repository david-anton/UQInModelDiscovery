import torch

from bayesianmdisc.models.isotropicmodellibrary import IsotropicModelLibrary

device = torch.device("cpu")


def get_expected_parameter_names() -> tuple[str, ...]:

    def get_expected_mr_parameter_names() -> tuple[str, ...]:
        return (
            "C_0_1 (MR)",
            "C_1_0 (NH)",
            "C_0_2",
            "C_1_1",
            "C_2_0",
            "C_0_3",
            "C_1_2",
            "C_2_1",
            "C_3_0",
        )

    def get_expected_ogden_parameter_names() -> tuple[str, ...]:
        parameter_names = []
        num_negative_terms = 4
        num_positive_terms = 4
        min_exponent = torch.tensor(-2.0)
        max_exponent = torch.tensor(2.0)
        negative_exponents = torch.linspace(min_exponent, 0.0, num_negative_terms + 1)[
            :-1
        ].tolist()
        positive_exponents = torch.linspace(0.0, max_exponent, num_positive_terms + 1)[
            1:
        ].tolist()
        exponents = negative_exponents + positive_exponents
        for index, exponent in zip(range(1, len(exponents) + 1), exponents):
            parameter_names += [f"O_{index} ({round(exponent,2)})"]
        return tuple(parameter_names)

    def get_expected_ln_feature_parameter_names() -> tuple[str, ...]:
        return ("ln_I2",)

    mr_parameter_names = get_expected_mr_parameter_names()
    ogden_parameter_names = get_expected_ogden_parameter_names()
    ln_feature_parameter_names = get_expected_ln_feature_parameter_names()
    return mr_parameter_names + ogden_parameter_names + ln_feature_parameter_names


def test_get_parameter_names() -> None:
    sut = IsotropicModelLibrary(device)

    actual = sut.parameter_names

    expected = get_expected_parameter_names()
    assert expected == actual
