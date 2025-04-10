import torch
import torch.nn as nn
from torch.func import grad, vmap

from bayesianmdisc.bayes.prior import PriorProtocol
from bayesianmdisc.customtypes import Device, Tensor, TorchLRScheduler, TorchOptimizer
from bayesianmdisc.data import DeformationInputs, TestCases
from bayesianmdisc.gppriors.priors import create_parameter_prior
from bayesianmdisc.gps import GaussianProcess
from bayesianmdisc.gps.base import GPMultivariateNormal
from bayesianmdisc.gps.multioutputgp import flatten_outputs
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import ModelProtocol
from bayesianmdisc.networks import FFNN
from bayesianmdisc.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_statistical_loss_history,
)

print_interval = 10
num_iters_lipschitz_pretraining = 2_000


def infer_gp_induced_prior(
    gp: GaussianProcess,
    model: ModelProtocol,
    prior_type: str,
    is_mean_trainable: bool,
    inputs: DeformationInputs,
    test_cases: TestCases,
    num_func_samples: int,
    resample: bool,
    num_iters_wasserstein: int,
    hiden_layer_size_lipschitz_nn: int,
    num_iters_lipschitz: int,
    lipschitz_func_pretraining: bool,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> PriorProtocol:
    output_dim = model.output_dim
    num_flattened_outputs = len(inputs) * output_dim

    penalty_coefficient_lipschitz = torch.tensor(10.0, device=device)
    learning_rate_lipschitz_func = 5e-4

    lr_decay_rate_prior = 1.0

    prior = create_parameter_prior(
        prior_type=prior_type,
        is_mean_trainable=is_mean_trainable,
        model_library=model,
        device=device,
    )
    lipschitz_func = FFNN(
        layer_sizes=[
            num_flattened_outputs,
            hiden_layer_size_lipschitz_nn,
            hiden_layer_size_lipschitz_nn,
            1,
        ],
        activation=nn.Softplus(),
        init_weights=nn.init.xavier_uniform_,
        init_bias=nn.init.zeros_,
    ).to(device)
    gp_distribution: GPMultivariateNormal = gp(inputs)

    if not resample:
        fixed_gp_func_values = gp_distribution.rsample(
            sample_shape=torch.Size([num_func_samples])
        )

    def freeze_gp(gp: GaussianProcess) -> None:
        gp.train(False)
        for parameters in gp.parameters():
            parameters.requires_grad = False
        likelihood = gp.likelihood
        for parameters in likelihood.parameters():
            parameters.requires_grad = False

    def create_prior_optimizer() -> TorchOptimizer:
        return torch.optim.RMSprop(params=prior.get_parameters_and_options())

    def create_lipschitz_func_optimizer() -> TorchOptimizer:
        return torch.optim.RMSprop(
            params=lipschitz_func.parameters(), lr=learning_rate_lipschitz_func
        )

    def create_learning_rate_scheduler(
        optimizer: TorchOptimizer, decay_rate: float
    ) -> TorchLRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    def draw_gp_func_values() -> Tensor:
        # Output of multi-output GP is already flattened, output = [output1_1:n, output2_1:n, ...].
        if resample:
            return gp_distribution.rsample(sample_shape=torch.Size([num_func_samples]))
        else:
            return fixed_gp_func_values

    def draw_model_func_values() -> Tensor:
        model_parameters = prior(num_func_samples)
        vmap_model_func = lambda _model_parameters: flatten_outputs(
            model(inputs, test_cases, _model_parameters)
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
        optimizer_lipschitz = create_lipschitz_func_optimizer()

        for iter_pretraining in range(num_iters_lipschitz_pretraining):
            gp_func_values = draw_gp_func_values()
            model_func_values = draw_model_func_values()

            optimizer_lipschitz.zero_grad(set_to_none=True)
            loss_lipschitz = lipschitz_func_loss(
                gp_func_values=gp_func_values,
                model_func_values=model_func_values,
                penalty_coefficient=penalty_coefficient_lipschitz,
            )
            loss_lipschitz.backward(retain_graph=True)
            optimizer_lipschitz.step()

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

    freeze_gp(gp)
    if lipschitz_func_pretraining:
        pretrain_lipschitz_func()

    optimizer_prior = create_prior_optimizer()
    optimizer_lipschitz = create_lipschitz_func_optimizer()
    lr_scheduler_prior = create_learning_rate_scheduler(
        optimizer_prior, lr_decay_rate_prior
    )
    wasserstein_loss_hist = []

    for iter_wasserstein in range(1, num_iters_wasserstein + 1):
        for _ in range(num_iters_lipschitz):
            gp_func_values = draw_gp_func_values()
            model_func_values = draw_model_func_values()

            optimizer_lipschitz.zero_grad(set_to_none=True)
            loss_lipschitz = lipschitz_func_loss(
                gp_func_values=gp_func_values,
                model_func_values=model_func_values,
                penalty_coefficient=penalty_coefficient_lipschitz,
            )
            loss_lipschitz.backward(retain_graph=True)
            optimizer_lipschitz.step()

        gp_func_values = draw_gp_func_values()
        model_func_values = draw_model_func_values()
        optimizer_prior.zero_grad(set_to_none=True)
        loss_wasserstein = wasserstein_loss(
            gp_func_values=gp_func_values, model_func_values=model_func_values
        )
        loss_wasserstein.backward(retain_graph=True)
        optimizer_prior.step()
        lr_scheduler_prior.step()
        loss_wasserstein_float = loss_wasserstein.detach().cpu().item()
        loss_lipschitz_float = loss_lipschitz.detach().cpu().item()
        wasserstein_loss_hist += [loss_wasserstein_float]
        print_progress(
            iter_wasserstein=iter_wasserstein,
            loss_wasserstein=loss_wasserstein_float,
            loss_lipschitz_func=loss_lipschitz_float,
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

    ############################################################
    prior.print_hyperparameters()
    ############################################################

    return prior.get_prior_distribution()
