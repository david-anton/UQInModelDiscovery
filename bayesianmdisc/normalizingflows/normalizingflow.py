from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, TypeAlias, Protocol

import normflows as nf
import torch

from bayesianmdisc.bayes.likelihood import LikelihoodProtocol
from bayesianmdisc.customtypes import (
    Device,
    NFBaseDistribution,
    NFNormalizingFlow,
    Tensor,
    TorchLRScheduler,
    TorchOptimizer,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.loaderssavers import PytorchModelLoader, PytorchModelSaver
from bayesianmdisc.normalizingflows.flows import (
    NormalizingFlow,
    NormalizingFlowProtocol,
    create_exponential_constrained_flow,
    create_masked_autoregressive_flow,
)
from bayesianmdisc.normalizingflows.target import (
    TargetDistributionWrapper,
)
from bayesianmdisc.normalizingflows.utility import freeze_model
from bayesianmdisc.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_statistical_loss_history,
)

NFNormalizingFlows: TypeAlias = list[NFNormalizingFlow]
ConstrainedOutputIndices: TypeAlias = list[int]
NormalizingFlowOutput: TypeAlias = NormalizingFlowProtocol

is_print_info_on = True
print_interval = 10
deactivation_initial_phase = 10_000
deactivation_interval = 1000
deactivation_threshold = 1e-12
num_deactivation_condition_samples = 4096

file_name_model = "normalizing_flow_parameters"


class VariationalInferencePriorProtocol(Protocol):
    dim: int

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob_with_grad(self, parameters: Tensor) -> Tensor:
        pass


PriorProtocol: TypeAlias = VariationalInferencePriorProtocol


@dataclass
class FitNormalizingFlowConfig:
    likelihood: LikelihoodProtocol
    prior: PriorProtocol
    num_flows: int
    relative_width_flow_layers: int
    num_samples: int
    initial_learning_rate: float
    final_learning_rate: float
    num_iterations: int
    deactivate_parameters: bool
    output_subdirectory: str
    project_directory: ProjectDirectory


def _create_base_distribution(
    num_parameters: int, device: Device
) -> NFBaseDistribution:
    return nf.distributions.base.DiagGaussian(num_parameters, trainable=False).to(
        device
    )


def _init_constrained_output_range(num_parameters: int) -> ConstrainedOutputIndices:
    return [_ for _ in range(num_parameters)]


def _create_normalizing_flow(
    relative_width_flow_layers: int,
    num_parameters: int,
    num_flows: int,
    base_distribution: NFBaseDistribution,
    device: Device,
) -> NormalizingFlow:

    width_layers = int(relative_width_flow_layers * num_parameters)
    constrained_output_indices = _init_constrained_output_range(num_parameters)
    flows = [
        create_masked_autoregressive_flow(num_parameters, width_layers)
        for _ in range(num_flows)
    ]
    flows += [
        create_exponential_constrained_flow(num_parameters, constrained_output_indices)
    ]
    return NormalizingFlow(flows, base_distribution, device).to(device)


def _fit_normalizing_flow(
    likelihood: LikelihoodProtocol,
    prior: PriorProtocol,
    num_flows: int,
    relative_width_flow_layers: int,
    num_samples: int,
    initial_learning_rate: float,
    final_learning_rate: float,
    num_iterations: int,
    deactivate_parameters: bool,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> NormalizingFlowOutput:
    num_parameters = prior.dim
    base_distribution = _create_base_distribution(num_parameters, device)
    target_distribution = TargetDistributionWrapper(likelihood, prior, device)

    def create_optimizer(parameters: Iterator[Tensor]) -> TorchOptimizer:
        return torch.optim.Adam(
            params=parameters, lr=initial_learning_rate, betas=(0.95, 0.999)
        )

    def create_exponential_learning_rate_scheduler(
        optimizer: TorchOptimizer,
    ) -> TorchLRScheduler:
        decay_rate = (final_learning_rate / initial_learning_rate) ** (
            1 / num_iterations
        )
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    def print_condition(iteration: int) -> bool:
        is_first = iteration == 1
        is_last = iteration == num_iterations
        is_interval_reached = iteration % print_interval == 0
        return is_first | is_last | is_interval_reached

    def parameter_deactivation_condition(iteration: int) -> bool:
        is_initial_phase_over = iteration >= deactivation_initial_phase
        is_interval_reached = iteration % deactivation_interval == 0
        return deactivate_parameters and is_initial_phase_over and is_interval_reached

    def draw_samples(num_samples: int) -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples)
        return list(samples)

    def train_normalizing_flow() -> None:
        optimizer = create_optimizer(normalizing_flow.parameters())
        lr_scheduler = create_exponential_learning_rate_scheduler(optimizer)

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

        def deactivate_parameters() -> None:
            samples_list = draw_samples(num_deactivation_condition_samples)
            samples = torch.stack(samples_list, dim=0)
            means = torch.mean(samples, dim=0)
            print(f"Parameter means: {means}")
            parameter_mask = means < deactivation_threshold
            indices = parameter_mask.nonzero().ravel().tolist()
            print(f"Masks model parameters with indices {indices}.")
            likelihood.model.deactivate_parameters(indices)

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

            if parameter_deactivation_condition(iteration):
                deactivate_parameters()

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

    def save_normalizing_flow() -> None:
        print("Save model ...")
        model_saver = PytorchModelSaver(project_directory)
        model_saver.save(normalizing_flow, file_name_model, output_subdirectory, device)

    normalizing_flow = _create_normalizing_flow(
        relative_width_flow_layers=relative_width_flow_layers,
        num_parameters=num_parameters,
        num_flows=num_flows,
        base_distribution=base_distribution,
        device=device,
    )

    print("Train normalizing flow ...")
    train_normalizing_flow()
    save_normalizing_flow()

    print("Postprocessing ...")
    freeze_model(normalizing_flow)

    return normalizing_flow


def fit_normalizing_flow(
    config: FitNormalizingFlowConfig, device: Device
) -> NormalizingFlowOutput:
    return _fit_normalizing_flow(
        likelihood=config.likelihood,
        prior=config.prior,
        num_flows=config.num_flows,
        relative_width_flow_layers=config.relative_width_flow_layers,
        num_samples=config.num_samples,
        initial_learning_rate=config.initial_learning_rate,
        final_learning_rate=config.final_learning_rate,
        num_iterations=config.num_iterations,
        deactivate_parameters=config.deactivate_parameters,
        output_subdirectory=config.output_subdirectory,
        project_directory=config.project_directory,
        device=device,
    )


@dataclass
class LoadNormalizingFlowConfig:
    num_parameters: int
    num_flows: int
    relative_width_flow_layers: int
    output_subdirectory: str
    project_directory: ProjectDirectory


def load_normalizing_flow(
    config: LoadNormalizingFlowConfig, device: Device
) -> NormalizingFlowProtocol:
    print("Load normalizing flow ...")
    base_distribution = _create_base_distribution(config.num_parameters, device)
    normalizing_flow = _create_normalizing_flow(
        relative_width_flow_layers=config.relative_width_flow_layers,
        num_parameters=config.num_parameters,
        num_flows=config.num_flows,
        base_distribution=base_distribution,
        device=device,
    )
    model_loader = PytorchModelLoader(config.project_directory)
    return model_loader.load(
        normalizing_flow, file_name_model, config.output_subdirectory
    )
