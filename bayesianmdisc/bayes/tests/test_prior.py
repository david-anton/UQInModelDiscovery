import pytest
import torch

from bayesianmdisc.bayes.prior import (
    create_independent_multivariate_normal_distributed_prior,
    create_independent_multivariate_studentT_distributed_prior,
    create_multivariate_normal_distributed_prior,
    create_multivariate_uniform_distributed_prior,
    create_univariate_gamma_distributed_prior,
    create_univariate_half_normal_distributed_prior,
    create_univariate_inverse_gamma_distributed_prior,
    create_univariate_normal_distributed_prior,
    create_univariate_uniform_distributed_prior,
    multiply_priors,
)
from bayesianmdisc.types import Tensor

device = torch.device("cpu")


def _expected_univariate_uniform_distributed_prior() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0]), torch.tensor(0.5)),
        (torch.tensor([-2.0]), torch.tensor(0.0)),
        (torch.tensor([2.0]), torch.tensor(0.0)),
    ]


def _expected_multivariate_uniform_distributed_prior() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0, 0.0]), torch.tensor(0.25)),
        (torch.tensor([-2.0, 0.0]), torch.tensor(0.0)),
        (torch.tensor([0.0, 2.0]), torch.tensor(0.0)),
    ]


def _expected_univariate_normal_distributed_prior() -> list[tuple[Tensor, Tensor]]:
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


def _expected_univariate_half_normal_distributed_prior() -> list[tuple[Tensor, Tensor]]:
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


def _expected_multivariate_normal_distributed_prior() -> list[tuple[Tensor, Tensor]]:
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


def _expected_independent_multivariate_normal_distributed_prior() -> (
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


def _expected_univariate_gamma_distributed_prior() -> list[tuple[Tensor, Tensor]]:
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


def _expected_univariate_inverse_gamma_distributed_prior() -> (
    list[tuple[Tensor, Tensor]]
):
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


def _expected_independent_multivariate_studentT_distributed_prior() -> (
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


def _expected_multiplied_prior() -> list[tuple[Tensor, Tensor]]:
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
    ("parameter", "expected"), _expected_univariate_uniform_distributed_prior()
)
def test_univariate_uniform_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    lower_limit = -1.0
    upper_limit = 1.0
    sut = create_univariate_uniform_distributed_prior(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_uniform_distributed_prior()
)
def test_multivariate_uniform_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])
    sut = create_multivariate_uniform_distributed_prior(
        lower_limits=lower_limits, upper_limits=upper_limits, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multivariate_uniform_distributed_prior_dimension() -> None:
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])
    sut = create_multivariate_uniform_distributed_prior(
        lower_limits=lower_limits, upper_limits=upper_limits, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_normal_distributed_prior()
)
def test_univariate_normal_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    mean = 0.0
    standard_deviation = 1.0
    sut = create_univariate_normal_distributed_prior(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_half_normal_distributed_prior()
)
def test_univariate_half_normal_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    standard_deviation = 1.0
    sut = create_univariate_half_normal_distributed_prior(
        standard_deviation=standard_deviation, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_normal_distributed_prior()
)
def test_multivariate_normal_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distributed_prior(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multivariate_normal_distributed_prior_dimension() -> None:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distributed_prior(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_normal_distributed_prior(),
)
def test_independent_multivariate_normal_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distributed_prior(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_normal_distributed_prior_dimension() -> None:
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distributed_prior(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_gamma_distributed_prior()
)
def test_univariate_gamma_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    concentration = 1.0
    rate = 1.0
    sut = create_univariate_gamma_distributed_prior(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_inverse_gamma_distributed_prior()
)
def test_univariate_inverse_gamma_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    concentration = 1.0
    rate = 1.0
    sut = create_univariate_inverse_gamma_distributed_prior(
        concentration=concentration, rate=rate, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_multiplied_prior(),
)
def test_multipled_priors(parameter: Tensor, expected: Tensor) -> None:
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_prior = create_univariate_uniform_distributed_prior(
        lower_limit=lower_bound_uniform_dist,
        upper_limit=upper_bound_uniform_dist,
        device=device,
    )
    mean_normal_dist = torch.tensor([0.0, 0.0])
    standard_deviations_normal_dist = torch.tensor([1.0, 1.0])
    normal_prior = create_independent_multivariate_normal_distributed_prior(
        means=mean_normal_dist,
        standard_deviations=standard_deviations_normal_dist,
        device=device,
    )

    sut = multiply_priors(priors=[uniform_prior, normal_prior])

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_multipled_priors_dimension() -> None:
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_prior = create_univariate_uniform_distributed_prior(
        lower_limit=lower_bound_uniform_dist,
        upper_limit=upper_bound_uniform_dist,
        device=device,
    )
    mean_normal_dist = torch.tensor([0.0, 0.0])
    standard_deviations_normal_dist = torch.tensor([1.0, 1.0])
    normal_prior = create_independent_multivariate_normal_distributed_prior(
        means=mean_normal_dist,
        standard_deviations=standard_deviations_normal_dist,
        device=device,
    )

    sut = multiply_priors(priors=[uniform_prior, normal_prior])

    actual = sut.dim
    expected = 3

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_independent_multivariate_studentT_distributed_prior(),
)
def test_independent_multivariate_studentT_distributed_prior(
    parameter: Tensor, expected: Tensor
) -> None:
    degrees_of_freedom = torch.tensor([1.0, 2.0])
    means = torch.tensor([0.0, 0.0])
    scales = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_studentT_distributed_prior(
        degrees_of_freedom=degrees_of_freedom, means=means, scales=scales, device=device
    )

    actual = sut.prob(parameter)

    torch.testing.assert_close(actual, expected)


def test_independent_multivariate_studentT_distributed_prior_dimension() -> None:
    degrees_of_freedom = torch.tensor([1.0, 2.0])
    means = torch.tensor([0.0, 0.0])
    scales = torch.tensor([1.0, 1.0])
    sut = create_independent_multivariate_studentT_distributed_prior(
        degrees_of_freedom=degrees_of_freedom, means=means, scales=scales, device=device
    )

    actual = sut.dim
    expected = 2

    torch.testing.assert_close(actual, expected)
