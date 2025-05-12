import pytest
import torch

from bayesianmdisc.bayes.distributions import (
    create_independent_multivariate_gamma_distribution,
    create_independent_multivariate_normal_distribution,
    create_independent_multivariate_studentT_distribution,
    create_multivariate_normal_distribution,
    create_multivariate_uniform_distribution,
    create_univariate_gamma_distribution,
    create_univariate_half_normal_distribution,
    create_univariate_inverse_gamma_distribution,
    create_univariate_normal_distribution,
    create_univariate_uniform_distribution,
    multiply_distributions,
)
from bayesianmdisc.customtypes import Tensor

device = torch.device("cpu")


def _expected_univariate_uniform_distribution() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0]), torch.tensor(0.5)),
        (torch.tensor([-2.0]), torch.tensor(0.0)),
        (torch.tensor([2.0]), torch.tensor(0.0)),
    ]


def _expected_multivariate_uniform_distribution() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0, 0.0]), torch.tensor(0.25)),
        (torch.tensor([-2.0, 0.0]), torch.tensor(0.0)),
        (torch.tensor([0.0, 2.0]), torch.tensor(0.0)),
    ]


def _expected_univariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    mean = torch.tensor(0.0)
    standard_deviation = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([0.0])
                )
            )[0],
        ),
        (
            torch.tensor([-1.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([-1.0])
                )
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([1.0])
                )
            )[0],
        ),
    ]


def _expected_univariate_half_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    standard_deviation = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.0]),
            torch.exp(
                torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                    torch.tensor([0.0])
                )
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.exp(
                torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                    torch.tensor([1.0])
                )
            )[0],
        ),
        (
            torch.tensor([2.0]),
            torch.exp(
                torch.distributions.HalfNormal(scale=standard_deviation).log_prob(
                    torch.tensor([2.0])
                )
            )[0],
        ),
    ]


def _expected_multivariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([0.0, 0.0]))
            ),
        ),
        (
            torch.tensor([-1.0, -1.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([-1.0, -1.0]))
            ),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([1.0, 1.0]))
            ),
        ),
    ]


def _expected_independent_multivariate_normal_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    mean = torch.tensor([0.0])
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.exp(
                3
                * torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([-1.0]))
            )[0],
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            torch.exp(
                3
                * torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.exp(
                (
                    torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([-1.0]))
                    + torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([0.0]))
                    + torch.distributions.Normal(
                        loc=mean, scale=standard_deviation
                    ).log_prob(torch.tensor([1.0]))
                )
            )[0],
        ),
    ]


def _expected_univariate_gamma_distribution() -> list[tuple[Tensor, Tensor]]:
    concentration = torch.tensor(1.0)
    rate = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.1]),
            torch.exp(
                torch.distributions.Gamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([0.1]))
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.exp(
                torch.distributions.Gamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
        (
            torch.tensor([2.0]),
            torch.exp(
                torch.distributions.Gamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([2.0]))
            )[0],
        ),
    ]


def _expected_independent_multivariate_gamma_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    concentrations = torch.tensor([0.5, 1.0, 2.0])
    rates = torch.tensor([1.0, 1.0, 1.0])
    return [
        (
            torch.tensor([0.1, 0.1, 0.1]),
            torch.exp(
                torch.sum(
                    torch.distributions.Gamma(
                        concentration=concentrations, rate=rates
                    ).log_prob(torch.tensor([0.1, 0.1, 0.1]))
                )
            ),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            torch.exp(
                torch.sum(
                    torch.distributions.Gamma(
                        concentration=concentrations, rate=rates
                    ).log_prob(torch.tensor([1.0, 1.0, 1.0]))
                )
            ),
        ),
        (
            torch.tensor([2.0, 2.0, 2.0]),
            torch.exp(
                torch.sum(
                    torch.distributions.Gamma(
                        concentration=concentrations, rate=rates
                    ).log_prob(torch.tensor([2.0, 2.0, 2.0]))
                )
            ),
        ),
    ]


def _expected_univariate_inverse_gamma_distribution() -> list[tuple[Tensor, Tensor]]:
    concentration = torch.tensor(1.0)
    rate = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.1]),
            torch.exp(
                torch.distributions.InverseGamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([0.1]))
            )[0],
        ),
        (
            torch.tensor([1.0]),
            torch.exp(
                torch.distributions.InverseGamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([1.0]))
            )[0],
        ),
        (
            torch.tensor([2.0]),
            torch.exp(
                torch.distributions.InverseGamma(
                    concentration=concentration, rate=rate
                ).log_prob(torch.tensor([2.0]))
            )[0],
        ),
    ]


def _expected_independent_multivariate_studentT_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    degrees_of_freedom = torch.tensor([1.0, 2.0])
    means = torch.tensor([0.0, 0.0])
    scales = torch.tensor([1.0, 1.0])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.exp(
                torch.sum(
                    torch.distributions.StudentT(
                        df=degrees_of_freedom, loc=means, scale=scales
                    ).log_prob(torch.tensor([0.0, 0.0]))
                )
            ),
        ),
        (
            torch.tensor([-1.0, -1.0]),
            torch.exp(
                torch.sum(
                    torch.distributions.StudentT(
                        df=degrees_of_freedom, loc=means, scale=scales
                    ).log_prob(torch.tensor([-1.0, -1.0]))
                )
            ),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.exp(
                torch.sum(
                    torch.distributions.StudentT(
                        df=degrees_of_freedom, loc=means, scale=scales
                    ).log_prob(torch.tensor([1.0, 1.0]))
                )
            ),
        ),
    ]


def _expected_multiplied_distributions() -> list[tuple[Tensor, Tensor]]:
    mean_normal_dist = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.eye(2)
    parameter_normal_dist = [0.0, 0.0]
    probability_normal_dist = torch.exp(
        torch.distributions.MultivariateNormal(
            loc=mean_normal_dist, covariance_matrix=covariance_matrix
        ).log_prob(torch.tensor(parameter_normal_dist))
    )

    return [
        (torch.tensor([0.0, *parameter_normal_dist]), 0.5 * probability_normal_dist),
        (torch.tensor([-2.0, *parameter_normal_dist]), 0.0 * probability_normal_dist),
        (torch.tensor([2.0, *parameter_normal_dist]), 0.0 * probability_normal_dist),
    ]


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_uniform_distribution()
)
def test_univariate_uniform_distribution(parameter: Tensor, expected: Tensor) -> None:
    lower_limit = -1.0
    upper_limit = 1.0
    sut = create_univariate_uniform_distribution(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_uniform_distribution()
)
def test_multivariate_uniform_distribution(parameter: Tensor, expected: Tensor) -> None:
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])
    sut = create_multivariate_uniform_distribution(
        lower_limits=lower_limits, upper_limits=upper_limits, device=device
    )

    actual = sut.prob(parameter)

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


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_normal_distribution()
)
def test_univariate_normal_distribution(parameter: Tensor, expected: Tensor) -> None:
    mean = 0.0
    standard_deviation = 1.0
    sut = create_univariate_normal_distribution(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


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

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_normal_distribution()
)
def test_multivariate_normal_distribution(parameter: Tensor, expected: Tensor) -> None:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distribution(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.prob(parameter)

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


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_normal_distribution(),
)
def test_independent_multivariate_normal_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distribution(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.prob(parameter)

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


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_gamma_distribution()
)
def test_univariate_gamma_distribution(parameter: Tensor, expected: Tensor) -> None:
    concentration = 1.0
    rate = 1.0
    sut = create_univariate_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_gamma_distribution(),
)
def test_independent_multivariate_gamma_distribuion(
    parameter: Tensor, expected: Tensor
) -> None:
    concentrations = torch.tensor([0.5, 1.0, 2.0])
    rates = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_gamma_distribution(
        concentrations=concentrations, rates=rates, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_gamma_distribution_dimension() -> None:
    concentrations = torch.tensor([0.5, 1.0, 2.0])
    rates = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_gamma_distribution(
        concentrations=concentrations, rates=rates, device=device
    )

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_inverse_gamma_distribution()
)
def test_univariate_inverse_gamma_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    concentration = 1.0
    rate = 1.0
    sut = create_univariate_inverse_gamma_distribution(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_multiplied_distributions(),
)
def test_multipled_distributions(parameter: Tensor, expected: Tensor) -> None:
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_distribution = create_univariate_uniform_distribution(
        lower_limit=lower_bound_uniform_dist,
        upper_limit=upper_bound_uniform_dist,
        device=device,
    )
    mean_normal_dist = torch.tensor([0.0, 0.0])
    standard_deviations_normal_dist = torch.tensor([1.0, 1.0])
    normal_distribution = create_independent_multivariate_normal_distribution(
        means=mean_normal_dist,
        standard_deviations=standard_deviations_normal_dist,
        device=device,
    )

    sut = multiply_distributions(
        distributions=[uniform_distribution, normal_distribution]
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multipled_distributions_dimension() -> None:
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_distribution = create_univariate_uniform_distribution(
        lower_limit=lower_bound_uniform_dist,
        upper_limit=upper_bound_uniform_dist,
        device=device,
    )
    mean_normal_dist = torch.tensor([0.0, 0.0])
    standard_deviations_normal_dist = torch.tensor([1.0, 1.0])
    normal_distribution = create_independent_multivariate_normal_distribution(
        means=mean_normal_dist,
        standard_deviations=standard_deviations_normal_dist,
        device=device,
    )

    sut = multiply_distributions(
        distributions=[uniform_distribution, normal_distribution]
    )

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_studentT_distribution(),
)
def test_independent_multivariate_studentT_distribution(
    parameter: Tensor, expected: Tensor
) -> None:
    degrees_of_freedom = torch.tensor([1.0, 2.0])
    means = torch.tensor([0.0, 0.0])
    scales = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom=degrees_of_freedom, means=means, scales=scales, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_studentT_distribution_dimension() -> None:
    degrees_of_freedom = torch.tensor([1.0, 2.0])
    means = torch.tensor([0.0, 0.0])
    scales = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_studentT_distribution(
        degrees_of_freedom=degrees_of_freedom, means=means, scales=scales, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)
