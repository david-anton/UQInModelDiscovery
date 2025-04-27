import os
from datetime import date
from typing import cast

import torch

from bayesianmdisc.bayes.likelihood import LikelihoodProtocol, create_likelihood
from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
    create_univariate_gamma_distributed_prior,
    multiply_priors,
)
from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.data import (
    DataReaderProtocol,
    KawabataDataReader,
    LinkaHeartDataReader,
    TreloarDataReader,
    data_set_label_kawabata,
    data_set_label_linka,
    data_set_label_treloar,
    determine_heteroscedastic_noise,
    split_data,
    validate_data,
)
from bayesianmdisc.gppriors import infer_gp_induced_prior
from bayesianmdisc.gps import (
    GP,
    GaussianProcess,
    IndependentMultiOutputGP,
    condition_gp,
    create_scaled_rbf_gaussian_process,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import (
    IsotropicModelLibrary,
    ModelProtocol,
    OrthotropicCANN,
    load_model_state,
    save_model_state,
    trim_model,
)
from bayesianmdisc.normalizingflows import (
    FitNormalizingFlowConfig,
    LoadNormalizingFlowConfig,
    NormalizingFlowProtocol,
    determine_statistical_moments,
    fit_normalizing_flow,
    load_normalizing_flow,
)
from bayesianmdisc.postprocessing.plot import (
    plot_histograms,
    plot_stresses_kawabata,
    plot_stresses_linka,
    plot_stresses_treloar,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)

data_set_label = data_set_label_treloar
use_gp_prior = True
retrain_normalizing_flow = True

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Input/output
current_date = date.today().strftime("%Y%m%d")
if data_set_label == data_set_label_treloar:
    input_directory = data_set_label
    data_reader: DataReaderProtocol = TreloarDataReader(
        input_directory, project_directory, device
    )
elif data_set_label == data_set_label_kawabata:
    input_directory = data_set_label
    data_reader = KawabataDataReader(input_directory, project_directory, device)
elif data_set_label == data_set_label_linka:
    input_directory = "heart_data_linka"
    data_reader = LinkaHeartDataReader(input_directory, project_directory, device)


if data_set_label == data_set_label_treloar:
    model: ModelProtocol = IsotropicModelLibrary(output_dim=1, device=device)
elif data_set_label == data_set_label_kawabata:
    model = IsotropicModelLibrary(output_dim=2, device=device)
elif data_set_label == data_set_label_linka:
    model = OrthotropicCANN(device)

prior_relative_noise_stddevs = 1e-1  # 5e-2
estimate_noise = True
min_noise_stddev = 1e-3
num_calibration_steps = 2
list_num_wasserstein_iterations = [20_000, 10_000]
list_relative_selection_thressholds = [2.0]
num_flows = 16
relative_width_flow_layers = 4
trim_metric = "mae"
num_samples_posterior = 4096


output_directory = f"{current_date}_{input_directory}_threshold_2_mae_estimatednoise"
output_subdirectory_name_prior = "prior"
output_subdirectory_name_posterior = "posterior"


def determine_number_of_parameters(model: ModelProtocol) -> int:
    num_noise_parameters = 1
    num_parameters = model.num_parameters
    if estimate_noise:
        num_parameters += num_noise_parameters
    return num_parameters


def determine_parameter_names(model: ModelProtocol) -> tuple[str, ...]:
    noise_parameter_name = ("rel. noise standard deviation",)
    parameter_names = model.parameter_names
    if estimate_noise:
        parameter_names = noise_parameter_name + parameter_names
    return parameter_names


def sample_from_posterior(
    normalizing_flow: NormalizingFlowProtocol,
) -> tuple[MomentsMultivariateNormal, NPArray]:

    def draw_samples() -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples_posterior)
        return list(samples)

    samples_list = draw_samples()
    return determine_statistical_moments(samples_list)


def plot_stresses(
    model: ModelProtocol,
    posterior_samples: NPArray,
    is_model_trimmed: bool,
    output_directory: str,
) -> None:
    def join_output_subdirectory() -> str:
        if is_model_trimmed:
            subdirectory_name = "trimmed_model"
        else:
            subdirectory_name = "untrimmed_model"
        return os.path.join(output_directory, subdirectory_name)

    output_subdirectory = join_output_subdirectory()

    def plot_treloar() -> None:
        plot_stresses_treloar(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

        def _plot_kawabata() -> None:
            input_directory = data_set_label_kawabata
            data_reader = KawabataDataReader(input_directory, project_directory, device)
            output_subdirectory_kawabata = os.path.join(output_subdirectory, "kawabata")

            inputs, test_cases, outputs = data_reader.read()
            isotropic_model = cast(IsotropicModelLibrary, model)
            isotropic_model.set_output_dimension(2)

            plot_stresses_kawabata(
                model=isotropic_model,
                parameter_samples=posterior_samples,
                inputs=inputs.detach().cpu().numpy(),
                outputs=outputs.detach().cpu().numpy(),
                test_cases=test_cases.detach().cpu().numpy(),
                output_subdirectory=output_subdirectory_kawabata,
                project_directory=project_directory,
                device=device,
            )

            isotropic_model.set_output_dimension(1)

        _plot_kawabata()

    def plot_kawabata() -> None:
        plot_stresses_kawabata(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    def plot_linka() -> None:
        plot_stresses_linka(
            model=cast(OrthotropicCANN, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    if data_set_label == data_set_label_treloar:
        plot_treloar()
    elif data_set_label == data_set_label_kawabata:
        plot_kawabata()
    elif data_set_label == data_set_label_linka:
        plot_linka()


inputs, test_cases, outputs = data_reader.read()
noise_stddevs = determine_heteroscedastic_noise(
    prior_relative_noise_stddevs, min_noise_stddev, outputs
)
validate_data(inputs, test_cases, outputs, noise_stddevs)

if use_gp_prior:
    splitted_data = split_data(
        data_set_label, inputs, test_cases, outputs, noise_stddevs
    )
    inputs_prior = splitted_data.inputs_prior
    inputs_posterior = splitted_data.inputs_posterior
    test_cases_prior = splitted_data.test_cases_prior
    test_cases_posterior = splitted_data.test_cases_posterior
    outputs_prior = splitted_data.outputs_prior
    outputs_posterior = splitted_data.outputs_posterior
    noise_stddevs_prior = splitted_data.noise_stddevs_prior
    noise_stddevs_posterior = splitted_data.noise_stddevs_posterior
else:
    inputs_posterior = inputs
    test_cases_posterior = test_cases
    outputs_posterior = outputs
    noise_stddevs_posterior = noise_stddevs

if retrain_normalizing_flow:
    for step in range(num_calibration_steps):
        is_last_step = step == num_calibration_steps - 1
        output_directory_step = os.path.join(
            output_directory, f"calibration_step_{step}"
        )
        num_parameters = determine_number_of_parameters(model)
        parameter_names = determine_parameter_names(model)

        def _create_likelihood() -> LikelihoodProtocol:
            if estimate_noise:
                relative_noise_stddevs = None
            else:
                relative_noise_stddevs = prior_relative_noise_stddevs
            return create_likelihood(
                model=model,
                relative_noise_stddev=relative_noise_stddevs,
                min_noise_stddev=min_noise_stddev,
                inputs=inputs_posterior,
                test_cases=test_cases_posterior,
                outputs=outputs_posterior,
                device=device,
            )

        def _determine_prior() -> PriorProtocol:
            def _determine_parameter_prior() -> PriorProtocol:
                def init_fixed_prior() -> PriorProtocol:
                    return create_independent_multivariate_gamma_distributed_prior(
                        concentrations=torch.tensor(
                            [0.1 for _ in range(num_parameters)], device=device
                        ),
                        rates=torch.tensor(
                            [10.0 for _ in range(num_parameters)], device=device
                        ),
                        device=device,
                    )

                def fit_gp_prior() -> PriorProtocol:
                    def create_gaussian_process() -> GaussianProcess:
                        is_single_outut_gp = output_dim == 1
                        jitter = 1e-7

                        def create_single_output_gp() -> GP:
                            gaussian_process = create_scaled_rbf_gaussian_process(
                                mean="zero",
                                input_dims=input_dim,
                                min_inputs=min_inputs,
                                max_inputs=max_inputs,
                                jitter=jitter,
                                device=device,
                            )
                            initial_parameters = torch.tensor(
                                [1.0] + [0.1 for _ in range(input_dim)], device=device
                            )
                            gaussian_process.set_parameters(initial_parameters)
                            return gaussian_process

                        def create_multi_output_gp() -> IndependentMultiOutputGP:
                            gaussian_processes = [
                                create_scaled_rbf_gaussian_process(
                                    mean="zero",
                                    input_dims=input_dim,
                                    min_inputs=min_inputs,
                                    max_inputs=max_inputs,
                                    jitter=jitter,
                                    device=device,
                                )
                                for _ in range(output_dim)
                            ]
                            initial_parameters = torch.tensor(
                                [1.0] + [0.1 for _ in range(input_dim)], device=device
                            )

                            for gaussian_process in gaussian_processes:
                                gaussian_process.set_parameters(initial_parameters)

                            return IndependentMultiOutputGP(
                                gps=tuple(gaussian_processes), device=device
                            )

                        if is_single_outut_gp:
                            return create_single_output_gp()
                        else:
                            return create_multi_output_gp()

                    def condition_gaussian_process(
                        inputs: Tensor, outputs: Tensor, noise_stddevs: Tensor
                    ) -> None:
                        condition_gp(
                            gaussian_process, inputs, outputs, noise_stddevs, device
                        )

                    def determine_prior_moments(
                        samples: Tensor,
                    ) -> tuple[MomentsMultivariateNormal, NPArray]:
                        samples_np = samples.detach().cpu().numpy()
                        moments = determine_moments_of_multivariate_normal_distribution(
                            samples_np
                        )
                        return moments, samples_np

                    output_subdirectory = os.path.join(
                        output_directory_step, output_subdirectory_name_prior
                    )
                    min_inputs = torch.amin(inputs, dim=0)
                    max_inputs = torch.amax(inputs, dim=0)
                    input_dim = inputs.size()[1]
                    output_dim = outputs.size()[1]

                    gaussian_process = create_gaussian_process()

                    optimize_gp_hyperparameters(
                        gaussian_process=gaussian_process,
                        inputs=inputs,
                        outputs=outputs,
                        initial_noise_stddevs=noise_stddevs,
                        num_iterations=int(5e4),
                        learning_rate=1e-3,
                        output_subdirectory=output_subdirectory,
                        project_directory=project_directory,
                        device=device,
                    )

                    condition_gaussian_process(
                        inputs_prior, outputs_prior, noise_stddevs_prior
                    )

                    gp_prior = infer_gp_induced_prior(
                        gp=gaussian_process,
                        model=model,
                        prior_type="inverse Gamma",
                        is_mean_trainable=True,
                        inputs=inputs,
                        test_cases=test_cases,
                        num_func_samples=32,
                        resample=True,
                        num_iters_wasserstein=list_num_wasserstein_iterations[step],
                        hiden_layer_size_lipschitz_nn=256,
                        num_iters_lipschitz=5,
                        lipschitz_func_pretraining=True,
                        output_subdirectory=output_subdirectory,
                        project_directory=project_directory,
                        device=device,
                    )

                    prior_samples = gp_prior.sample(num_samples=4096)
                    prior_moments, prior_samples_np = determine_prior_moments(
                        prior_samples
                    )

                    plot_histograms(
                        parameter_names=model.parameter_names,
                        true_parameters=tuple(None for _ in range(num_parameters)),
                        moments=prior_moments,
                        samples=prior_samples_np,
                        algorithm_name="gp_prior",
                        output_subdirectory=output_subdirectory,
                        project_directory=project_directory,
                    )
                    return gp_prior

                if use_gp_prior:
                    return fit_gp_prior()
                else:
                    return init_fixed_prior()

            def _init_relative_noise_stddev_prior() -> PriorProtocol:
                return create_univariate_gamma_distributed_prior(
                    concentration=0.1,
                    rate=10.0,
                    device=device,
                )

            prior_parameters = _determine_parameter_prior()
            if estimate_noise:
                prior_relative_noise_stddev = _init_relative_noise_stddev_prior()
                return multiply_priors([prior_relative_noise_stddev, prior_parameters])
            else:
                return prior_parameters

        likelihood = _create_likelihood()
        prior = _determine_prior()

        fit_normalizing_flow_config = FitNormalizingFlowConfig(
            likelihood=likelihood,
            prior=prior,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            num_samples=64,
            initial_learning_rate=5e-4,
            final_learning_rate=1e-4,
            num_iterations=100_000,
            output_subdirectory=output_directory_step,
            project_directory=project_directory,
        )

        normalizing_flow = fit_normalizing_flow(fit_normalizing_flow_config, device)
        posterior_moments, posterior_samples = sample_from_posterior(normalizing_flow)

        output_subdirectory_posterior = os.path.join(
            output_directory_step, output_subdirectory_name_posterior
        )
        plot_histograms(
            parameter_names=parameter_names,
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=posterior_moments,
            samples=posterior_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_posterior,
            project_directory=project_directory,
        )
        plot_stresses(
            model=model,
            posterior_samples=posterior_samples,
            is_model_trimmed=False,
            output_directory=output_directory_step,
        )

        if not is_last_step:
            trim_model(
                model=model,
                metric=trim_metric,
                relative_thresshold=list_relative_selection_thressholds[step],
                parameter_samples=torch.from_numpy(posterior_samples)
                .type(torch.get_default_dtype())
                .to(device),
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
            )
            plot_stresses(
                model=model,
                posterior_samples=posterior_samples,
                is_model_trimmed=True,
                output_directory=output_directory_step,
            )
            model.reduce_to_activated_parameters()

        save_model_state(model, output_directory_step, project_directory)
else:
    for step in range(num_calibration_steps):
        is_first_step = step == 0
        is_last_step = step == num_calibration_steps - 1
        output_directory_step = os.path.join(
            output_directory, f"calibration_step_{step}"
        )
        if is_first_step:
            input_directory_step = output_directory_step
        else:
            input_step = step - 1
            input_directory_step = os.path.join(
                output_directory, f"calibration_step_{input_step}"
            )

        if not is_first_step:
            load_model_state(model, input_directory_step, project_directory, device)

        num_parameters = determine_number_of_parameters(model)
        parameter_names = determine_parameter_names(model)

        load_normalizing_flow_config = LoadNormalizingFlowConfig(
            num_parameters=num_parameters,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            output_subdirectory=output_directory_step,
            project_directory=project_directory,
        )
        normalizing_flow = load_normalizing_flow(load_normalizing_flow_config, device)
        posterior_moments, posterior_samples = sample_from_posterior(normalizing_flow)

        output_subdirectory_posterior = os.path.join(
            output_directory_step, output_subdirectory_name_posterior
        )
        plot_histograms(
            parameter_names=parameter_names,
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=posterior_moments,
            samples=posterior_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_posterior,
            project_directory=project_directory,
        )
        plot_stresses(
            model=model,
            posterior_samples=posterior_samples,
            is_model_trimmed=False,
            output_directory=output_directory_step,
        )

        if not is_last_step:
            trim_model(
                model=model,
                metric=trim_metric,
                relative_thresshold=list_relative_selection_thressholds[step],
                parameter_samples=torch.from_numpy(posterior_samples)
                .type(torch.get_default_dtype())
                .to(device),
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
            )
            plot_stresses(
                model=model,
                posterior_samples=posterior_samples,
                is_model_trimmed=True,
                output_directory=output_directory_step,
            )
