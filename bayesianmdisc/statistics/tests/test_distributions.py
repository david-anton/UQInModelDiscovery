import pytest
import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.statistics.distributions import (
    create_independent_multivariate_gamma_distribution,
    create_independent_multivariate_half_normal_distribution,
    create_independent_multivariate_normal_distribution,
    create_independent_multivariate_studentT_distribution,
    create_multivariate_normal_distribution,
    create_multivariate_uniform_distribution,
    create_univariate_gamma_distribution,
    create_univariate_half_normal_distribution,
    create_univariate_inverse_gamma_distribution,
    create_univariate_normal_distribution,
    create_univariate_uniform_distribution,
)

device = torch.device("cpu")


# univariate uniform
def _expected_univariate_uniform_distribution() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0]), torch.log(torch.tensor(0.5))),
        (torch.tensor([-2.0]), torch.log(torch.tensor(0.0))),
        (torch.tensor([2.0]), torch.log(torch.tensor(0.0))),
    ]


# multivariate uniform
def _expected_multivariate_uniform_distribution() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0, 0.0]), torch.log(torch.tensor(0.25))),
        (torch.tensor([-2.0, 0.0]), torch.log(torch.tensor(0.0))),
        (torch.tensor([0.0, 2.0]), torch.log(torch.tensor(0.0))),
    ]


# univariate normal
def _expected_univariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    mean = torch.tensor(0.0)
    standard_deviation = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                torch.tensor([0.0])
            )[0],
        ),
        (
            torch.tensor([-1.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                torch.tensor([-1.0])
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                torch.tensor([1.0])
            )[0],
        ),
    ]


# univariate half normal
def _expected_univariate_half_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    standard_deviation = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.0]),
            torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                torch.tensor([0.0])
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                torch.tensor([1.0])
            )[0],
        ),
    ]


# multivariate normal
def _expected_multivariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            ).log_prob(torch.tensor([0.0, 0.0])),
        ),
        (
            torch.tensor([-1.0, -1.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            ).log_prob(torch.tensor([-1.0, -1.0])),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            ).log_prob(torch.tensor([1.0, 1.0])),
        ),
    ]


# independent multivariate normal
def _expected_independent_multivariate_normal_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    mean = torch.tensor([0.0])
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            3
            * torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                torch.tensor([-1.0])
            )[0],
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            3
            * torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                torch.tensor([1.0])
            )[0],
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            (
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([-1.0])
                )
                + torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([0.0]))
                + torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
    ]


def _expected_independent_multivariate_normal_distribution_individual() -> (
    list[tuple[Tensor, Tensor]]
):
    mean = torch.tensor([0.0])
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor(
                [
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([-1.0]))[0]
                ]
            ).repeat(
                3,
            ),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([1.0]))[0]
                ],
            ).repeat(
                3,
            ),
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([-1.0]))[0],
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([0.0]))[0],
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([1.0]))[0],
                ],
            ),
        ),
    ]


# independent multivariate half normal
def _expected_independent_multivariate_half_normal_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([0.0, 0.0]),
            2
            * torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                torch.tensor([0.0])
            )[0],
        ),
        (
            torch.tensor([1.0, 1.0]),
            2
            * torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                torch.tensor([1.0])
            )[0],
        ),
        (
            torch.tensor([0.0, 1.0]),
            (
                torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                    torch.tensor([0.0])
                )
                + torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                    torch.tensor([1.0])
                )
            )[0],
        ),
    ]


def _expected_independent_multivariate_half_normal_distribution_individual() -> (
    list[tuple[Tensor, Tensor]]
):
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor(
                [
                    torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                        torch.tensor([0.0])
                    )[0]
                ]
            ).repeat(
                2,
            ),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                        torch.tensor([1.0])
                    )[0]
                ],
            ).repeat(
                2,
            ),
        ),
        (
            torch.tensor([0.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                        torch.tensor([0.0])
                    )[0],
                    torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                        torch.tensor([1.0])
                    )[0],
                ],
            ),
        ),
    ]


# univariate gamma
def _expected_univariate_gamma_distribution() -> list[tuple[Tensor, Tensor]]:
    concentration = torch.tensor(2.0)
    rate = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.1]),
            torch.distributions.Gamma(concentration=concentration, rate=rate).log_prob(
                torch.tensor([0.1])
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.distributions.Gamma(concentration=concentration, rate=rate).log_prob(
                torch.tensor([1.0])
            )[0],
        ),
    ]


# independent multivariate gamma
def _expected_independent_multivariate_gamma_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    concentration = torch.tensor([2.0])
    rate = torch.tensor([1.0])
    return [
        (
            torch.tensor([0.1, 0.1]),
            2
            * torch.distributions.Gamma(
                concentration=concentration, rate=rate
            ).log_prob(torch.tensor([0.1]))[0],
        ),
        (
            torch.tensor([1.0, 1.0]),
            2
            * torch.distributions.Gamma(
                concentration=concentration, rate=rate
            ).log_prob(torch.tensor([1.0]))[0],
        ),
        (
            torch.tensor([0.1, 1.0]),
            (
                torch.distributions.Gamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([0.1]))
                + torch.distributions.Gamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
    ]


def _expected_independent_multivariate_gamma_distribution_individual() -> (
    list[tuple[Tensor, Tensor]]
):
    concentration = torch.tensor([2.0])
    rate = torch.tensor([1.0])
    return [
        (
            torch.tensor([0.1, 0.1]),
            torch.tensor(
                [
                    torch.distributions.Gamma(
                        concentration=concentration, rate=rate
                    ).log_prob(torch.tensor([0.1]))[0]
                ]
            ).repeat(
                2,
            ),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.Gamma(
                        concentration=concentration, rate=rate
                    ).log_prob(torch.tensor([1.0]))[0]
                ],
            ).repeat(
                2,
            ),
        ),
        (
            torch.tensor([0.1, 1.0]),
            torch.tensor(
                [
                    torch.distributions.Gamma(
                        concentration=concentration, rate=rate
                    ).log_prob(torch.tensor([0.1]))[0],
                    torch.distributions.Gamma(
                        concentration=concentration, rate=rate
                    ).log_prob(torch.tensor([1.0]))[0],
                ],
            ),
        ),
    ]


# univariate inverse gamma
def _expected_univariate_inverse_gamma_distribution() -> list[tuple[Tensor, Tensor]]:
    concentration = torch.tensor(2.0)
    rate = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.1]),
            torch.distributions.InverseGamma(
                concentration=concentration, rate=rate
            ).log_prob(torch.tensor([0.1]))[0],
        ),
        (
            torch.tensor([1.0]),
            torch.distributions.InverseGamma(
                concentration=concentration, rate=rate
            ).log_prob(torch.tensor([1.0]))[0],
        ),
    ]


# independent multivariate normal
def _expected_independent_multivariate_studentT_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    degrees_of_freedom = torch.tensor([2.0])
    mean = torch.tensor([0.0])
    scales = torch.tensor([0.1])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            3
            * torch.distributions.StudentT(
                df=degrees_of_freedom,
                loc=mean,
                scale=scales,
            ).log_prob(torch.tensor([-1.0]))[0],
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            3
            * torch.distributions.StudentT(
                df=degrees_of_freedom,
                loc=mean,
                scale=scales,
            ).log_prob(torch.tensor([1.0]))[0],
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            (
                torch.distributions.StudentT(
                    df=degrees_of_freedom,
                    loc=mean,
                    scale=scales,
                ).log_prob(torch.tensor([-1.0]))
                + torch.distributions.StudentT(
                    df=degrees_of_freedom,
                    loc=mean,
                    scale=scales,
                ).log_prob(torch.tensor([0.0]))
                + torch.distributions.StudentT(
                    df=degrees_of_freedom,
                    loc=mean,
                    scale=scales,
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
    ]


def _expected_independent_multivariate_studentT_distribution_individual() -> (
    list[tuple[Tensor, Tensor]]
):
    degrees_of_freedom = torch.tensor([2.0])
    mean = torch.tensor([0.0])
    scales = torch.tensor([0.1])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor(
                [
                    torch.distributions.StudentT(
                        df=degrees_of_freedom,
                        loc=mean,
                        scale=scales,
                    ).log_prob(torch.tensor([-1.0]))[0]
                ]
            ).repeat(
                3,
            ),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.StudentT(
                        df=degrees_of_freedom,
                        loc=mean,
                        scale=scales,
                    ).log_prob(torch.tensor([1.0]))[0]
                ],
            ).repeat(
                3,
            ),
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor(
                [
                    torch.distributions.StudentT(
                        df=degrees_of_freedom,
                        loc=mean,
                        scale=scales,
                    ).log_prob(torch.tensor([-1.0]))[0],
                    torch.distributions.StudentT(
                        df=degrees_of_freedom,
                        loc=mean,
                        scale=scales,
                    ).log_prob(torch.tensor([0.0]))[0],
                    torch.distributions.StudentT(
                        df=degrees_of_freedom,
                        loc=mean,
                        scale=scales,
                    ).log_prob(torch.tensor([1.0]))[0],
                ],
            ),
        ),
    ]


################################################################################
################################################################################


# univariate uniform
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_uniform_distribution()
)
def test_univariate_uniform_distribution(parameter: Tensor, expected: Tensor) -> None:
    lower_limit = -1.0
    upper_limit = 1.0
    sut = create_univariate_uniform_distribution(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_univariate_uniform_distribution_dimension() -> None:
    lower_limit = -1.0
    upper_limit = 1.0
    sut = create_univariate_uniform_distribution(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut.dim
    expected = 1

    torch.testing.assert_close(actual, expected)


# multivariate uniform
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_uniform_distribution()
)
def test_multivariate_uniform_distribution(parameter: Tensor, expected: Tensor) -> None:
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])
    sut = create_multivariate_uniform_distribution(
        lower_limits=lower_limits, upper_limits=upper_limits, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multivariate_uniform_distribution_dimension() -> None:
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])
    sut = create_multivariate_uniform_distribution(
        lower_limits=lower_limits, upper_limits=upper_limits, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


# univariate normal
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_normal_distribution()
)
def test_univariate_normal_distribution(parameter: Tensor, expected: Tensor) -> None:
    mean = 0.0
    standard_deviation = 1.0
    sut = create_univariate_normal_distribution(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_univariate_normal_distribution_dimension() -> None:
    mean = 0.0
    standard_deviation = 1.0
    sut = create_univariate_normal_distribution(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut.dim
    expected = 1

    torch.testing.assert_close(actual, expected)


# univariate half normal
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_half_normal_distribution()
)
def test_univariate_half_normal_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    standard_deviation = 1.0
    sut = create_univariate_half_normal_distribution(
        standard_deviation=standard_deviation, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_univariate_half_normal_distribution_dimension() -> None:
    standard_deviation = 1.0
    sut = create_univariate_half_normal_distribution(
        standard_deviation=standard_deviation, device=device
    )

    actual = sut.dim
    expected = 1

    torch.testing.assert_close(actual, expected)


# multivariate normal
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_normal_distribution()
)
def test_multivariate_normal_distribution(parameter: Tensor, expected: Tensor) -> None:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distribution(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multivariate_normal_distribution_dimension() -> None:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distribution(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


# independent multivariate normal
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_independent_multivariate_normal_distribution()
)
def test_independent_multivariate_normal_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distribution(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_normal_distribution_individual(),
)
def test_independent_multivariate_normal_distribution_individual(
    parameter: Tensor, expected: Tensor
) -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distribution(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.log_probs_individual(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_normal_distribution_dimension() -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distribution(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)


# independent multivariate half normal
@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_half_normal_distribution(),
)
def test_independent_multivariate_half_normal_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    standard_deviations = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_half_normal_distribution(
        standard_deviations=standard_deviations, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_half_normal_distribution_individual(),
)
def test_independent_multivariate_half_normal_distribution_individual(
    parameter: Tensor, expected: Tensor
) -> None:
    standard_deviations = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_half_normal_distribution(
        standard_deviations=standard_deviations, device=device
    )

    actual = sut.log_probs_individual(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_half_normal_distribution_dimension() -> None:
    standard_deviations = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_half_normal_distribution(
        standard_deviations=standard_deviations, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


# univariate gamma
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_gamma_distribution()
)
def test_univariate_gamma_distribution(parameter: Tensor, expected: Tensor) -> None:
    concentration = 2.0
    rate = 1.0
    sut = create_univariate_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_univariate_gamma_distribution_dimension() -> None:
    concentration = 2.0
    rate = 1.0
    sut = create_univariate_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.dim
    expected = 1

    torch.testing.assert_close(actual, expected)


# independent multivariate gamma
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_independent_multivariate_gamma_distribution()
)
def test_independent_multivariate_gamma_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    concentrations = torch.tensor([2.0, 2.0])
    rates = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_gamma_distribution(
        concentrations=concentrations, rates=rates, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_gamma_distribution_individual(),
)
def test_independent_multivariate_gamma_distribution_individual(
    parameter: Tensor, expected: Tensor
) -> None:
    concentrations = torch.tensor([2.0, 2.0])
    rates = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_gamma_distribution(
        concentrations=concentrations, rates=rates, device=device
    )

    actual = sut.log_probs_individual(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_gamma_distribution_dimension() -> None:
    concentrations = torch.tensor([2.0, 2.0])
    rates = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_gamma_distribution(
        concentrations=concentrations, rates=rates, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


# univariate inverse gamma
@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_inverse_gamma_distribution()
)
def test_univariate_inverse_gamma_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    concentration = 2.0
    rate = 1.0
    sut = create_univariate_inverse_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_univariate_inverse_gamma_distribution_dimension() -> None:
    concentration = 2.0
    rate = 1.0
    sut = create_univariate_inverse_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.dim
    expected = 1

    torch.testing.assert_close(actual, expected)


# independent multivariate studentT
@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_studentT_distribution(),
)
def test_independent_multivariate_studentT_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    degrees_of_freedom = torch.tensor([2.0, 2.0, 2.0])
    means = torch.tensor([0.0, 0.0, 0.0])
    scales = torch.tensor([0.1, 0.1, 0.1])
    sut = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom=degrees_of_freedom,
        means=means,
        scales=scales,
        device=device,
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_studentT_distribution_individual(),
)
def test_independent_multivariate_studentT_distribution_individual(
    parameter: Tensor, expected: Tensor
) -> None:
    degrees_of_freedom = torch.tensor([2.0, 2.0, 2.0])
    means = torch.tensor([0.0, 0.0, 0.0])
    scales = torch.tensor([0.1, 0.1, 0.1])
    sut = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom=degrees_of_freedom,
        means=means,
        scales=scales,
        device=device,
    )

    actual = sut.log_probs_individual(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_studentT_distribution_dimension() -> None:
    degrees_of_freedom = torch.tensor([2.0, 2.0, 2.0])
    means = torch.tensor([0.0, 0.0, 0.0])
    scales = torch.tensor([0.1, 0.1, 0.1])
    sut = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom=degrees_of_freedom,
        means=means,
        scales=scales,
        device=device,
    )

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)
