import torch

from bayesianmdisc.models.isotropicmodellibrary import IsotropicModelLibrary

device = torch.device("cpu")


def get_expected_parameter_names() -> tuple[str, ...]:

    def get_expected_ogden_parameter_names() -> tuple[str, ...]:
        parameter_names = []
        num_terms = 33
        min_exponent = torch.tensor(-4.0)
        max_exponent = torch.tensor(4.0)
        exponents = torch.linspace(min_exponent, max_exponent, num_terms).tolist()
        for index, exponent in zip(range(1, num_terms + 1), exponents):
            parameter_names += [f"O_{index} (exponent: {round(exponent,2)})"]
        return tuple(parameter_names)

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

    ogden_parameter_names = get_expected_ogden_parameter_names()
    mr_parameter_names = get_expected_mr_parameter_names()
    return ogden_parameter_names + mr_parameter_names


def test_get_parameter_names() -> None:
    sut = IsotropicModelLibrary(device)

    actual = sut.get_parameter_names()

    expected = get_expected_parameter_names()
    assert expected == actual
