from typing import Optional, TypeAlias

import gpytorch
import torch

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.errors import GPError

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
GPMultivariateNormalList: TypeAlias = list[GPMultivariateNormal]
NamedParameters: TypeAlias = dict[str, Tensor]
GaussianLikelihood: TypeAlias = gpytorch.likelihoods.GaussianLikelihood
GaussianLikelihoodList: TypeAlias = list[GaussianLikelihood]
TrainingDataTuple: TypeAlias = tuple[Optional[Tensor], ...]
MarginalLogLikelihood: TypeAlias = gpytorch.mlls.MarginalLogLikelihood


def validate_training_data(
    inputs: TrainingDataTuple, outputs: TrainingDataTuple, num_outputs: int
) -> None:
    num_input_sets = len(inputs)
    num_output_sets = len(outputs)
    if num_input_sets != num_output_sets:
        raise GPError(
            f"""Number of input and output data sets is expected to be the same
            but is {num_input_sets} and {num_output_sets}"""
        )
    elif num_input_sets != num_outputs or num_output_sets != num_outputs:
        raise GPError(
            f"""Number of input and output data sets is expected to be the same 
            as number of outputs but is {num_input_sets} and {num_output_sets} 
            and number of outputs is {num_outputs}"""
        )


def validate_likelihood_noise_variance(
    noise_variance: Tensor, num_outputs: int
) -> None:
    if not noise_variance.ndim == 1:
        raise GPError(
            """Noise standard deviation tensor is expected 
            to be a one-dimensional tensor."""
        )
    if torch.numel(noise_variance) != num_outputs:
        raise GPError(
            """Noise standard deviation tensor is expected 
            to have one entry for each GP output."""
        )
