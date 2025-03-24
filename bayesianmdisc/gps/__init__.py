from typing import TypeAlias

from bayesianmdisc.gps.gp import (
    GP,
    create_scaled_matern_gaussian_process,
    create_scaled_rbf_gaussian_process,
)
from bayesianmdisc.gps.multioutputgp import IndependentMultiOutputGP
from bayesianmdisc.gps.training import condition_gp, optimize_gp_hyperparameters

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP

__all__ = [
    "GaussianProcess",
    "create_scaled_matern_gaussian_process",
    "create_scaled_rbf_gaussian_process",
    "IndependentMultiOutputGP",
    "condition_gp",
    "optimize_gp_hyperparameters",
]
