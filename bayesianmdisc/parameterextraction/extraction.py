from typing import cast, TypeAlias
import math
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
file_name_model_parameters_nf = "normalizing_flow_parameters"


FuncSizes: TypeAlias = list[int]
TensorList: TypeAlias = list[Tensor]


class LipschitzFunction(nn.Module):
    def __init__(
        self,
        func_sizes: FuncSizes,
        num_layers: int,
        rel_layer_width: float,
        device: Device,
    ) -> None:
        super().__init__()
        self.func_sizes = func_sizes
        self._num_funcs = len(func_sizes)
        self._num_layers = num_layers
        self._rel_layer_width = rel_layer_width
        self._device = device
        self._networks = nn.ModuleList(self._init_networks())

    def __call__(self, all_funcs: TensorList) -> TensorList:
        return [network(funcs) for network, funcs in zip(self._networks, all_funcs)]

    def forward_single_network(self, funcs: Tensor, func_dim: int) -> Tensor:
        return self._networks[func_dim](funcs)

    def _init_networks(self) -> list[Module]:
        def init_one_network(func_size: int) -> Module:
            hidden_layer_size = int(math.floor(self._rel_layer_width * func_size))
            layer_sizes = [func_size]
            layer_sizes += [hidden_layer_size for _ in range(self._num_layers)]
            layer_sizes += [1]
            return FFNN(
                layer_sizes=layer_sizes,
                activation=nn.Softplus(),
                init_weights=nn.init.xavier_uniform_,
                init_bias=nn.init.zeros_,
                use_spectral_norm=False,
            ).to(self._device)

        return [init_one_network(func_size) for func_size in self.func_sizes]


def extract_gp_inducing_parameter_distribution(
    gp: GaussianProcess,
    model: ModelProtocol,
    output_selector: OutputSelectorProtocol,
    distribution_type: str,
    inputs: Tensor,
    test_cases: TestCases,
    num_func_samples: int,
    lipschitz_penalty_coefficient: float,
    num_iters_wasserstein: int,
    num_layers_lipschitz_nn: int,
    relative_width_lipschitz_nn: float,
    num_iters_lipschitz: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> DistributionProtocol:
    lipschitz_coefficient = torch.tensor(lipschitz_penalty_coefficient, device=device)
    initial_lr_lipschitz_func = 1e-4
    lr_decay_rate_lipschitz = 1.0
    initial_lr_distribution = 5e-4
    lr_decay_rate_distribution = 0.9999

    def determine_func_dims() -> int:
        return gp.num_gps

    def determine_func_sizes() -> FuncSizes:
        output_mask = output_selector.selection_mask
        splitted_output_mask = torch.chunk(output_mask, func_dims)
        return [int(torch.sum(mask).item()) for mask in splitted_output_mask]

    def freeze_gp(gp: GaussianProcess) -> None:
        gp.train(False)
        for parameters in gp.parameters():
            parameters.requires_grad = False
        likelihood = gp.likelihood
        for parameters in likelihood.parameters():
            parameters.requires_grad = False

    def infer_gp_distribution() -> GPMultivariateNormal:
        return gp(inputs)

    def create_distribution_optimizer() -> TorchOptimizer:
        return torch.optim.RMSprop(
            params=distribution.parameters(),
            lr=initial_lr_distribution,
        )

    def create_lipschitz_func_optimizer() -> TorchOptimizer:
        return torch.optim.AdamW(
            params=lipschitz_func.parameters(),
            lr=initial_lr_lipschitz_func,
            betas=(0.0, 0.9),
        )

    def create_lr_scheduler(
        optimizer: TorchOptimizer, decay_rate: float
    ) -> TorchLRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    def draw_gp_funcs() -> Tensor:
        func_samples = gp_distribution.rsample(
            sample_shape=torch.Size([num_func_samples])
        )
        vmap_func = lambda _func_sample: output_selector(_func_sample)
        return vmap(vmap_func)(func_samples)

    def draw_model_funcs() -> Tensor:
        model_parameters = distribution(num_func_samples)
        vmap_func = lambda _model_parameters: output_selector(
            flatten_outputs(model(inputs, test_cases, _model_parameters))
        )
        return vmap(vmap_func)(model_parameters)

    def split_in_func_dimensions(all_funcs: Tensor) -> TensorList:
        return list(torch.split(all_funcs, list(func_sizes), dim=1))

    def draw_splitted_gp_outputs() -> TensorList:
        all_funcs = draw_gp_funcs()
        return split_in_func_dimensions(all_funcs)

    def draw_splitted_model_outputs() -> TensorList:
        all_funcs = draw_model_funcs()
        return split_in_func_dimensions(all_funcs)

    def wasserstein_losses(
        all_gp_funcs: TensorList, all_model_funcs: TensorList
    ) -> Tensor:
        lipschitz_outputs_gp = lipschitz_func(all_gp_funcs)
        lipschitz_outputs_model = lipschitz_func(all_model_funcs)
        return torch.concat(
            [
                torch.unsqueeze(
                    torch.mean(lipschitz_output_gp - lipschitz_output_model), dim=0
                )
                for lipschitz_output_gp, lipschitz_output_model in zip(
                    lipschitz_outputs_gp, lipschitz_outputs_model
                )
            ]
        )

    def wasserstein_loss(
        all_gp_funcs: TensorList, all_model_funcs: TensorList
    ) -> Tensor:
        losses = wasserstein_losses(all_gp_funcs, all_model_funcs)
        return torch.sum(losses)

    def gradient_penalty(
        all_gp_funcs: TensorList, all_model_funcs: TensorList
    ) -> Tensor:
        def gradient_penalty_for_one_output_dim(
            gp_funcs: Tensor, model_funcs: Tensor, func_dim: int
        ) -> Tensor:
            one = torch.tensor(1.0, device=device)
            single_lipschitz_func = lambda func: lipschitz_func.forward_single_network(
                func, func_dim
            )[0]

            def l2_norm(grads: Tensor) -> Tensor:
                return torch.sqrt(torch.sum(torch.square(grads)))

            epsilon = torch.rand((num_func_samples, 1), device=device)
            combined_funcs = epsilon * model_funcs + (one - epsilon) * gp_funcs

            vmap_grad_penalty = lambda _func: (
                torch.square(
                    l2_norm(grad(single_lipschitz_func, argnums=0)(_func)) - one
                )
            )
            return torch.mean(vmap(vmap_grad_penalty)(combined_funcs))

        return torch.concat(
            [
                torch.unsqueeze(
                    gradient_penalty_for_one_output_dim(
                        gp_output, model_output, func_dim
                    ),
                    dim=0,
                )
                for gp_output, model_output, func_dim in zip(
                    all_gp_funcs, all_model_funcs, range(func_dims)
                )
            ]
        )

    def lipschitz_losses(
        all_gp_funcs: TensorList, all_model_funcs: TensorList
    ) -> Tensor:
        _wasserstein_losses = wasserstein_loss(all_gp_funcs, all_model_funcs)
        _gradient_penalty = gradient_penalty(all_gp_funcs, all_model_funcs)
        return -_wasserstein_losses + lipschitz_coefficient * _gradient_penalty

    def lipschitz_loss(all_gp_funcs: TensorList, all_model_funcs: TensorList) -> Tensor:
        losses = lipschitz_losses(all_gp_funcs, all_model_funcs)
        return torch.sum(losses)

    def print_condition(iter_wasserstein: int) -> bool:
        is_first = iter_wasserstein == 1
        is_last = iter_wasserstein == num_iters_wasserstein
        is_interval_reached = iter_wasserstein % print_interval == 0
        return is_first | is_last | is_interval_reached

    def print_progress(
        iter_wasserstein: int, loss_wasserstein: float, loss_lipschitz: float
    ) -> None:
        if print_condition(iter_wasserstein):
            print("############################################################")
            print(f"Iteration: {iter_wasserstein}")
            print(f"Loss Wasserstein distance: {loss_wasserstein}")
            print(f"Loss Lipschitz function: {loss_lipschitz}")

    func_dims = determine_func_dims()
    func_sizes = determine_func_sizes()

    freeze_gp(gp)
    gp_distribution = infer_gp_distribution()
    distribution = create_parameter_distribution(
        distribution_type=distribution_type,
        is_mean_trainable=True,
        model=model,
        device=device,
    )
    lipschitz_func = LipschitzFunction(
        func_sizes=func_sizes,
        num_layers=num_layers_lipschitz_nn,
        rel_layer_width=relative_width_lipschitz_nn,
        device=device,
    )

    optimizer_distribution = create_distribution_optimizer()
    optimizer_lipschitz = create_lipschitz_func_optimizer()
    lr_scheduler_distribution = create_lr_scheduler(
        optimizer_distribution, lr_decay_rate_distribution
    )
    lr_scheduler_lipschitz = create_lr_scheduler(
        optimizer_lipschitz, lr_decay_rate_lipschitz
    )
    wasserstein_loss_hist = []

    for iter_wasserstein in range(1, num_iters_wasserstein + 1):
        for _ in range(num_iters_lipschitz):
            gp_outputs = draw_splitted_gp_outputs()
            model_outputs = draw_splitted_model_outputs()

            def lipschitz_loss_closure() -> float:
                optimizer_lipschitz.zero_grad(set_to_none=True)
                loss_lipschitz = lipschitz_loss(gp_outputs, model_outputs)
                loss_lipschitz.backward(retain_graph=True)
                return loss_lipschitz.item()

            loss_lipschitz = optimizer_lipschitz.step(lipschitz_loss_closure)

        gp_outputs = draw_splitted_gp_outputs()
        model_outputs = draw_splitted_model_outputs()

        def wasserstein_loss_closure() -> float:
            optimizer_distribution.zero_grad(set_to_none=True)
            loss_wasserstein = wasserstein_loss(gp_outputs, model_outputs)
            loss_wasserstein.backward(retain_graph=True)
            return loss_wasserstein.item()

        loss_wasserstein = optimizer_distribution.step(wasserstein_loss_closure)
        lr_scheduler_distribution.step()
        lr_scheduler_lipschitz.step()
        wasserstein_loss_hist += [loss_wasserstein]
        print_progress(
            iter_wasserstein=iter_wasserstein,
            loss_wasserstein=loss_wasserstein,
            loss_lipschitz=loss_lipschitz,
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
