from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, TypeAlias

import normflows as nf
import torch

from bayesianmdisc.bayes.likelihood import LikelihoodProtocol
from bayesianmdisc.bayes.prior import PriorProtocol
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.normalizingflows.flows import (
    NormalizingFlow,
    create_exponential_constrained_flow,
    create_masked_autoregressive_flow,
)
from bayesianmdisc.normalizingflows.postprocessing import determine_statistical_moments
from bayesianmdisc.normalizingflows.target import (
    TargetDistributionWrapper,
)
from bayesianmdisc.normalizingflows.utility import freeze_model
from bayesianmdisc.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_statistical_loss_history,
)
from bayesianmdisc.statistics.utility import MomentsMultivariateNormal
from bayesianmdisc.customtypes import (
    Device,
    NFNormalizingFlow,
    NPArray,
    Tensor,
    TorchLRScheduler,
    TorchOptimizer,
)

NFNormalizingFlows: TypeAlias = list[NFNormalizingFlow]
ConstrainedOutputIndices: TypeAlias = list[int]
NormalizingFlowOutput: TypeAlias = tuple[MomentsMultivariateNormal, NPArray]

is_print_info_on = True
print_interval = 10
num_samples_output = int(1e4)


@dataclass
class NormalizingFlowConfig:
    likelihood: LikelihoodProtocol
    prior: PriorProtocol
    num_flows: int
    relative_width_flow_layers: int
    num_samples: int
    learning_rate: float
    learning_rate_decay_rate: float
    num_iterations: int
    output_subdirectory: str
    project_directory: ProjectDirectory


def _fit_normalizing_flow(
    likelihood: LikelihoodProtocol,
    prior: PriorProtocol,
    num_flows: int,
    relative_width_flow_layers: int,
    num_samples: int,
    learning_rate: float,
    learning_rate_decay_rate: float,
    num_iterations: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> NormalizingFlowOutput:
    num_parameters = prior.dim
    base_distribution = nf.distributions.base.DiagGaussian(
        num_parameters, trainable=False
    ).to(device)
    target_distribution = TargetDistributionWrapper(likelihood, prior, device)

    def create_normalizing_flow() -> NormalizingFlow:
        width_layers = int(relative_width_flow_layers * num_parameters)
        constrained_output_indices = init_constrained_output_range()
        flows = [
            create_masked_autoregressive_flow(num_parameters, width_layers)
            for _ in range(num_flows)
        ]
        flows += [
            create_exponential_constrained_flow(
                num_parameters, constrained_output_indices
            )
        ]
        return NormalizingFlow(flows, base_distribution, device).to(device)

    def init_constrained_output_range() -> ConstrainedOutputIndices:
        return [_ for _ in range(num_parameters)]

    def create_optimizer(parameters: Iterator[Tensor]) -> TorchOptimizer:
        return torch.optim.Adam(
            params=parameters, lr=learning_rate, betas=(0.95, 0.999)
        )

    def create_exponential_learning_rate_scheduler(
        optimizer: TorchOptimizer, decay_rate: float
    ) -> TorchLRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    def print_condition(iteration: int) -> bool:
        is_first = iteration == 1
        is_last = iteration == num_iterations
        is_interval_reached = iteration % print_interval == 0
        return is_first | is_last | is_interval_reached

    def train_normalizing_flow() -> None:
        optimizer = create_optimizer(normalizing_flow.parameters())
        lr_scheduler = create_exponential_learning_rate_scheduler(
            optimizer, learning_rate_decay_rate
        )

        def reverse_kld_func(samples_base: Tensor, log_probs_base: Tensor) -> Tensor:
            u = samples_base
            log_prob_u = log_probs_base
            x, sum_log_det_u = normalizing_flow.forward_u_and_sum_log_det_u(u)
            log_prob_x = target_distribution.log_prob(x)

            return torch.mean(log_prob_u - sum_log_det_u - log_prob_x, dim=0)

        def run_iteration() -> Tensor:
            optimizer.zero_grad()
            samples_base, log_probs_base = base_distribution(num_samples)
            kld = reverse_kld_func(samples_base, log_probs_base)
            kld.backward(retain_graph=True)
            optimizer.step()
            lr_scheduler.step()
            return kld.detach().cpu()

        print("############################################################")
        print(f"Start training ...")
        time_total_start = perf_counter()
        kld_hist: list[float] = []
        for iteration in range(1, num_iterations + 1):
            time_iteration_start = perf_counter()

            kld = run_iteration()
            kld_hist += [kld.item()]

            if print_condition(iteration):
                time_iteration_end = perf_counter()
                time_iteration = time_iteration_end - time_iteration_start
                print(f"Iteration: {iteration}")
                print(f"KLD: {kld.item()}")
                print(f"Time iteration: {time_iteration}")

        print("############################################################")

        # Postprocessing
        history_plotter_config = HistoryPlotterConfig()
        plot_statistical_loss_history(
            loss_hist=kld_hist,
            statistical_quantity="KLD",
            file_name=f"loss.png",
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            config=history_plotter_config,
        )

        print("Training of normalizing flow finished.")
        time_total_end = perf_counter()
        time_total = time_total_end - time_total_start
        print(f"Total training time: {time_total}")

    normalizing_flow = create_normalizing_flow()
    train_normalizing_flow()

    print("Postprocessing ...")
    freeze_model(normalizing_flow)

    def draw_samples(num_samples: int) -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples)
        return list(samples)

    samples_list = draw_samples(num_samples_output)
    moments, samples = determine_statistical_moments(samples_list)
    return moments, samples


def fit_normalizing_flow(
    config: NormalizingFlowConfig, device: Device
) -> NormalizingFlowOutput:
    return _fit_normalizing_flow(
        likelihood=config.likelihood,
        prior=config.prior,
        num_flows=config.num_flows,
        relative_width_flow_layers=config.relative_width_flow_layers,
        num_samples=config.num_samples,
        learning_rate=config.learning_rate,
        learning_rate_decay_rate=config.learning_rate_decay_rate,
        num_iterations=config.num_iterations,
        output_subdirectory=config.output_subdirectory,
        project_directory=config.project_directory,
        device=device,
    )
