from torch.func import vmap

from bayesianmdisc.bayes.likelihood import Likelihood
from bayesianmdisc.bayes.prior import Prior
from bayesianmdisc.types import Device, Tensor


class TargetDistributionWrapper:
    def __init__(self, likelihood: Likelihood, prior: Prior, device: Device) -> None:
        self.device = device
        self._likelihood = likelihood
        self._prior = prior

    def log_prob(self, parameters: Tensor) -> Tensor:

        def vmap_log_prob_func(_parameters: Tensor) -> Tensor:
            log_prob_likelihood = self._likelihood.log_prob_with_grad(_parameters)
            log_prob_prior = self._prior.log_prob_with_grad(_parameters)
            return log_prob_likelihood + log_prob_prior

        if parameters.dim() == 1:
            return vmap_log_prob_func(parameters)
        else:
            return vmap(vmap_log_prob_func, randomness="same")(parameters)
