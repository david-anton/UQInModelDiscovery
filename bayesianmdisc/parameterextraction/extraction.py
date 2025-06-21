from typing import cast

import torch
import torch.nn as nn
from torch.func import grad, vmap

from bayesianmdisc.bayes.distributions import DistributionProtocol
from bayesianmdisc.customtypes import (
    Device,
    Module,
    Tensor,
    TorchLRScheduler,
    TorchOptimizer,
)
from bayesianmdisc.gps import GaussianProcess
from bayesianmdisc.gps.base import GPMultivariateNormal
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.loaderssavers import PytorchModelLoader, PytorchModelSaver
from bayesianmdisc.models import ModelProtocol, OutputSelectorProtocol
from bayesianmdisc.networks.ffnn import FFNN
from bayesianmdisc.normalizingflows import (
    NormalizingFlowDistribution,
    NormalizingFlowProtocol,
)
from bayesianmdisc.parameterextraction.parameterdistributions import (
    NormalizingFlowParameterDistribution,
    create_parameter_distribution,
)
from bayesianmdisc.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_statistical_loss_history,
)
from bayesianmdisc.testcases import TestCases
from bayesianmdisc.utility import flatten_outputs

print_interval = 10
num_iters_lipschitz_pretraining = 2_000
file_name_model_parameters_nf = "normalizing_flow_parameters"


def extract_gp_inducing_parameter_distribution(
    gp: GaussianProcess,
    model: ModelProtocol,
    output_selector: OutputSelectorProtocol,
    distribution_type: str,
    is_mean_trainable: bool,
    inputs: Tensor,
    test_cases: TestCases,
    num_func_samples: int,
    resample: bool,
    lipschitz_penalty_coefficient: float,
    num_iters_wasserstein: int,
    num_layers_lipschitz_nn: int,
    layer_size_lipschitz_nn: int,
    num_iters_lipschitz: int,
    lipschitz_func_pretraining: bool,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> DistributionProtocol:
    num_flattened_outputs = output_selector.total_num_selected_outputs

    penalty_coefficient_lipschitz = torch.tensor(
        lipschitz_penalty_coefficient, device=device
    )
    initial_learning_rate_lipschitz_func = 1e-4
    lr_decay_rate_lipschitz_func = 1.0
    # final_learning_rate_lipschitz_func = 1e-4
    # lr_decay_rate_lipschitz_func = (
    #     final_learning_rate_lipschitz_func / initial_learning_rate_lipschitz_func
    # ) ** (1 / num_iters_wasserstein)

    initial_learning_rate_distribution = 5e-4
    lr_decay_rate_distribution = 0.9999
    # final_learning_rate_distribution = 1e-6
    # lr_decay_rate_distribution = (
    #     final_learning_rate_distribution / initial_learning_rate_distribution
    # ) ** (1 / num_iters_wasserstein)

    def create_lipschitz_network(
        num_layers: int, layer_size: int, device: Device
    ) -> Module:
        layer_sizes = [num_flattened_outputs]
        layer_sizes += [layer_size for _ in range(num_layers)]
        layer_sizes += [1]
        return FFNN(
            layer_sizes=layer_sizes,
            activation=nn.SiLU(),  # nn.LeakyReLU(),
            init_weights=nn.init.xavier_uniform_,
            init_bias=nn.init.zeros_,
            use_layer_norm=False,
        ).to(device)

    def freeze_gp(gp: GaussianProcess) -> None:
        gp.train(False)
        for parameters in gp.parameters():
            parameters.requires_grad = False
        likelihood = gp.likelihood
        for parameters in likelihood.parameters():
            parameters.requires_grad = False

    def create_distribution_optimizer() -> TorchOptimizer:
        return torch.optim.RMSprop(
            params=distribution.parameters(),
            lr=initial_learning_rate_distribution,
        )

    def create_lipschitz_func_optimizer() -> TorchOptimizer:
        return torch.optim.AdamW(
            params=lipschitz_func.parameters(),
            lr=initial_learning_rate_lipschitz_func,
            betas=(0.0, 0.9),
        )

    def create_learning_rate_scheduler(
        optimizer: TorchOptimizer, decay_rate: float
    ) -> TorchLRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    def sample_from_gp() -> Tensor:
        # Output of multi-output GP is already flattened, output = [output1_1:n, output2_1:n, ...].
        func_samples = gp_distribution.rsample(
            sample_shape=torch.Size([num_func_samples])
        )
        vmap_func = lambda _func_sample: output_selector(_func_sample)
        return vmap(vmap_func)(func_samples)

    def draw_gp_func_values() -> Tensor:
        if resample:
            return sample_from_gp()
        else:
            return fixed_gp_func_values

    def draw_model_func_values() -> Tensor:
        model_parameters = distribution(num_func_samples)
        vmap_model_func = lambda _model_parameters: output_selector(
            flatten_outputs(model(inputs, test_cases, _model_parameters))
        )
        return vmap(vmap_model_func)(model_parameters)

    def wasserstein_loss(gp_func_values: Tensor, model_func_values: Tensor) -> Tensor:
        lipschitz_func_gp = lipschitz_func(gp_func_values)
        lipschitz_func_model = lipschitz_func(model_func_values)
        return torch.mean(lipschitz_func_gp - lipschitz_func_model)

    def lipschitz_func_loss(
        gp_func_values: Tensor, model_func_values: Tensor, penalty_coefficient: Tensor
    ) -> Tensor:
        def gradient_penalty(
            gp_func_values: Tensor, model_func_values: Tensor
        ) -> Tensor:
            one = torch.tensor(1.0, device=device)
            _lipschitz_func = lambda func_values: lipschitz_func(func_values)[0]

            def l2_norm(values: Tensor) -> Tensor:
                return torch.sqrt(torch.sum(torch.square(values)))

            epsilon = torch.rand((num_func_samples, 1), device=device)
            combined_funcs = (
                epsilon * model_func_values + (one - epsilon) * gp_func_values
            )

            vmap_grad_penalty = lambda func_values: (
                torch.square(
                    l2_norm(grad(_lipschitz_func, argnums=0)(func_values)) - one
                )
            )
            return torch.mean(vmap(vmap_grad_penalty)(combined_funcs))

        return -wasserstein_loss(
            gp_func_values, model_func_values
        ) + penalty_coefficient * gradient_penalty(gp_func_values, model_func_values)

    def pretrain_lipschitz_func() -> None:
        print("Start pretraining of Lipschitz function ...")
        optimizer_lipschitz_func = create_lipschitz_func_optimizer()

        for iter_pretraining in range(num_iters_lipschitz_pretraining):
            gp_func_values = draw_gp_func_values()
            model_func_values = draw_model_func_values()

            def lipschitz_loss_closure() -> float:
                optimizer_lipschitz_func.zero_grad(set_to_none=True)
                loss_lipschitz = lipschitz_func_loss(
                    gp_func_values,
                    model_func_values,
                    penalty_coefficient_lipschitz,
                )
                loss_lipschitz.backward(retain_graph=True)
                return loss_lipschitz.item()

            loss_lipschitz = optimizer_lipschitz_func.step(lipschitz_loss_closure)

            if print_condition(iter_pretraining):
                print(f"Lipschitz loss: {loss_lipschitz}")

    def print_condition(iter_wasserstein: int) -> bool:
        is_first = iter_wasserstein == 1
        is_last = iter_wasserstein == num_iters_wasserstein
        is_interval_reached = iter_wasserstein % print_interval == 0
        return is_first | is_last | is_interval_reached

    def print_progress(
        iter_wasserstein: int, loss_wasserstein: float, loss_lipschitz_func: float
    ) -> None:
        if print_condition(iter_wasserstein):
            print("############################################################")
            print(f"Iteration: {iter_wasserstein}")
            print(f"Loss Wasserstein distance: {loss_wasserstein}")
            print(f"Loss Lipschitz function: {loss_lipschitz_func}")

    distribution = create_parameter_distribution(
        distribution_type=distribution_type,
        is_mean_trainable=is_mean_trainable,
        model=model,
        device=device,
    )
    lipschitz_func = create_lipschitz_network(
        num_layers=num_layers_lipschitz_nn,
        layer_size=layer_size_lipschitz_nn,
        device=device,
    )

    gp_distribution: GPMultivariateNormal = gp(inputs)

    if not resample:
        fixed_gp_func_values = sample_from_gp()

    freeze_gp(gp)
    if lipschitz_func_pretraining:
        pretrain_lipschitz_func()

    optimizer_distribution = create_distribution_optimizer()
    optimizer_lipschitz_func = create_lipschitz_func_optimizer()
    lr_scheduler_distribution = create_learning_rate_scheduler(
        optimizer_distribution, lr_decay_rate_distribution
    )
    lr_scheduler_lipschitz_func = create_learning_rate_scheduler(
        optimizer_lipschitz_func, lr_decay_rate_lipschitz_func
    )
    wasserstein_loss_hist = []

    for iter_wasserstein in range(1, num_iters_wasserstein + 1):
        for _ in range(num_iters_lipschitz):
            gp_func_values = draw_gp_func_values()
            model_func_values = draw_model_func_values()

            def lipschitz_loss_closure() -> float:
                optimizer_lipschitz_func.zero_grad(set_to_none=True)
                loss_lipschitz = lipschitz_func_loss(
                    gp_func_values,
                    model_func_values,
                    penalty_coefficient_lipschitz,
                )
                loss_lipschitz.backward(retain_graph=True)
                return loss_lipschitz.item()

            loss_lipschitz = optimizer_lipschitz_func.step(lipschitz_loss_closure)

        gp_func_values = draw_gp_func_values()
        model_func_values = draw_model_func_values()

        def wasserstein_loss_closure() -> float:
            optimizer_distribution.zero_grad(set_to_none=True)
            loss_wasserstein = wasserstein_loss(gp_func_values, model_func_values)
            loss_wasserstein.backward(retain_graph=True)
            return loss_wasserstein.item()

        loss_wasserstein = optimizer_distribution.step(wasserstein_loss_closure)
        lr_scheduler_distribution.step()
        lr_scheduler_lipschitz_func.step()
        wasserstein_loss_hist += [loss_wasserstein]
        print_progress(
            iter_wasserstein=iter_wasserstein,
            loss_wasserstein=loss_wasserstein,
            loss_lipschitz_func=loss_lipschitz,
        )

    history_plotter_config = HistoryPlotterConfig()
    plot_statistical_loss_history(
        loss_hist=wasserstein_loss_hist,
        statistical_quantity="W1",
        file_name="loss_w1.png",
        output_subdir=output_subdirectory,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    distribution.print_hyperparameters()
    return distribution.get_distribution()


def save_normalizing_flow_parameter_distribution(
    distribution: NormalizingFlowDistribution,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    print("Save normalizing flow parameter distribution ...")
    normalizing_flow = distribution.normalizing_flow
    model_saver = PytorchModelSaver(project_directory)
    model_saver.save(
        cast(Module, normalizing_flow),
        file_name_model_parameters_nf,
        output_subdirectory,
        device,
    )


def load_normalizing_flow_parameter_distribution(
    model: ModelProtocol,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> NormalizingFlowDistribution:
    print("Load normalizing flow parameter distribution ...")
    model_loader = PytorchModelLoader(project_directory)
    _distribution = cast(
        NormalizingFlowParameterDistribution,
        create_parameter_distribution(
            distribution_type="normalizing flow",
            is_mean_trainable=True,
            model=model,
            device=device,
        ),
    )
    distribution = cast(NormalizingFlowDistribution, _distribution.get_distribution())
    normalizing_flow = distribution.normalizing_flow
    loaded_normalizing_flow = model_loader.load(
        cast(Module, normalizing_flow),
        file_name_model_parameters_nf,
        output_subdirectory,
    )
    distribution.normalizing_flow = cast(
        NormalizingFlowProtocol, loaded_normalizing_flow
    )
    return distribution
