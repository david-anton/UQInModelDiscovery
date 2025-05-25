import os
from datetime import date
from typing import cast

import torch

from bayesianmdisc.bayes.distributions import (
    DistributionProtocol,
    sample_and_analyse_distribution,
)
from bayesianmdisc.customtypes import NPArray
from bayesianmdisc.data import (
    DataSetProtocol,
    KawabataDataSet,
    LinkaHeartDataSet,
    TreloarDataSet,
    data_set_label_kawabata,
    data_set_label_linka,
    data_set_label_treloar,
    determine_heteroscedastic_noise,
    validate_data,
)
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
    select_model_through_sobol_sensitivity_analysis,
)
from bayesianmdisc.parameterextraction import (
    extract_gp_inducing_parameter_distribution,
    load_normalizing_flow_parameter_distribution,
)
from bayesianmdisc.postprocessing.plot import (
    plot_gp_stresses_treloar,
    plot_histograms,
    plot_model_stresses_kawabata,
    plot_model_stresses_linka,
    plot_model_stresses_treloar,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed

data_set_label = data_set_label_treloar
retrain_posterior = True

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
    data_set: DataSetProtocol = TreloarDataSet(
        input_directory, project_directory, device
    )
    model: ModelProtocol = IsotropicModelLibrary(output_dim=1, device=device)
elif data_set_label == data_set_label_kawabata:
    input_directory = data_set_label
    data_set = KawabataDataSet(input_directory, project_directory, device)
    model = IsotropicModelLibrary(output_dim=2, device=device)
elif data_set_label == data_set_label_linka:
    input_directory = "heart_data_linka"
    data_set = LinkaHeartDataSet(input_directory, project_directory, device)
    model = OrthotropicCANN(device)

relative_noise_stddevs = 5e-2
min_absolute_noise_stddev = 5e-2
list_num_wasserstein_iterations = [50_000, 50_000]
num_samples_parameter_distribution = 8192
num_samples_factor_sensitivity_analysis = 4096
first_sobol_index_thresshold = 1e-6


output_directory = f"{current_date}_{input_directory}_normalizingflow_relnoise5e-2_minabsnoise5e-2_lipschitz_iters5_lambda10_lr1_samples16_layer2_width512_numinputs32_sobolsensitivities"
output_subdirectory_name_gp = "gp"
output_subdirectory_name_parameters = "parameters"
output_subdirectory_name_sensitivities = "sensitivity_analysis"


def plot_gp_stresses(
    gaussian_process: GaussianProcess,
    output_subdirectory: str,
) -> None:

    def plot_treloar() -> None:
        plot_gp_stresses_treloar(
            gaussian_process=gaussian_process,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    if data_set_label == data_set_label_treloar:
        plot_treloar()


def plot_model_stresses(
    model: ModelProtocol,
    model_parameter_samples: NPArray,
    subdirectory_name: str,
    output_directory: str,
) -> None:

    def join_output_subdirectory() -> str:
        return os.path.join(output_directory, subdirectory_name)

    output_subdirectory = join_output_subdirectory()

    def plot_treloar() -> None:
        plot_model_stresses_treloar(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=model_parameter_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

        def _plot_kawabata() -> None:
            input_directory = data_set_label_kawabata
            data_set = KawabataDataSet(input_directory, project_directory, device)
            output_subdirectory_kawabata = os.path.join(output_subdirectory, "kawabata")

            inputs, test_cases, outputs = data_set.read_data()
            isotropic_model = cast(IsotropicModelLibrary, model)
            isotropic_model.set_output_dimension(2)

            plot_model_stresses_kawabata(
                model=isotropic_model,
                parameter_samples=model_parameter_samples,
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
        plot_model_stresses_kawabata(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=model_parameter_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    def plot_linka() -> None:
        plot_model_stresses_linka(
            model=cast(OrthotropicCANN, model),
            parameter_samples=model_parameter_samples,
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


inputs, test_cases, outputs = data_set.read_data()
noise_stddevs = determine_heteroscedastic_noise(
    relative_noise_stddevs, min_absolute_noise_stddev, outputs
)
validate_data(inputs, test_cases, outputs, noise_stddevs)
num_discovery_steps = len(list_num_wasserstein_iterations)

if retrain_posterior:
    for step in range(num_discovery_steps):
        is_first_step = step == 0
        output_directory_step = os.path.join(output_directory, f"discovery_step_{step}")
        output_subdirectory_gp = os.path.join(
            output_directory_step, output_subdirectory_name_gp
        )
        output_subdirectory_parameters = os.path.join(
            output_directory_step, output_subdirectory_name_parameters
        )
        output_subdirectory_sensitivities = os.path.join(
            output_directory_step, output_subdirectory_name_sensitivities
        )

        def create_gp() -> GaussianProcess:
            min_inputs = torch.amin(inputs, dim=0)
            max_inputs = torch.amax(inputs, dim=0)
            input_dim = inputs.size()[1]
            output_dim = outputs.size()[1]
            is_single_outut_gp = output_dim == 1
            jitter = 1e-7

            def create_single_output_gp() -> GP:
                gp_mean = "zero"
                gaussian_process = create_scaled_rbf_gaussian_process(
                    mean=gp_mean,
                    input_dim=input_dim,
                    min_inputs=min_inputs,
                    max_inputs=max_inputs,
                    jitter=jitter,
                    device=device,
                )
                initial_parameters_output_scale = [1.0]
                initial_parameters_length_scale = [0.1 for _ in range(input_dim)]
                initial_parameters_kernel = (
                    initial_parameters_output_scale + initial_parameters_length_scale
                )

                if gp_mean == "linear":
                    initial_parameters_weights = [1.0 for _ in range(input_dim)]
                    initial_parameters_bias = [0.0]
                    initial_parameters_mean = (
                        initial_parameters_weights + initial_parameters_bias
                    )
                    initial_parameters = torch.tensor(
                        initial_parameters_mean + initial_parameters_kernel,
                        device=device,
                    )
                else:
                    initial_parameters = torch.tensor(
                        initial_parameters_kernel, device=device
                    )
                gaussian_process.set_parameters(initial_parameters)
                return gaussian_process

            def create_independent_multi_output_gp() -> IndependentMultiOutputGP:
                gaussian_processes = [
                    create_single_output_gp() for _ in range(output_dim)
                ]
                return IndependentMultiOutputGP(
                    gps=tuple(gaussian_processes), device=device
                )

            if is_single_outut_gp:
                return create_single_output_gp()
            else:
                return create_independent_multi_output_gp()

        def select_gp_prior() -> None:
            optimize_gp_hyperparameters(
                gaussian_process=gaussian_process,
                inputs=inputs,
                outputs=outputs,
                initial_noise_stddevs=noise_stddevs,
                num_iterations=int(5e4),
                learning_rate=1e-3,
                output_subdirectory=output_subdirectory_gp,
                project_directory=project_directory,
                device=device,
            )

        def infer_gp_posterior() -> None:
            condition_gp(
                gaussian_process=gaussian_process,
                inputs=inputs,
                outputs=outputs,
                noise_stddevs=noise_stddevs,
                device=device,
            )
            plot_gp_stresses(
                gaussian_process=gaussian_process,
                output_subdirectory=output_subdirectory_gp,
            )

        def extract_parameter_distribution() -> DistributionProtocol:
            _data_set = cast(TreloarDataSet, data_set)
            inputs_extraction, test_cases_extraction = (
                _data_set.generate_uniform_inputs(num_points_per_test_case=16)
            )
            return extract_gp_inducing_parameter_distribution(
                gp=gaussian_process,
                model=model,
                distribution_type="normalizing flow",
                is_mean_trainable=True,
                inputs=inputs_extraction,
                test_cases=test_cases_extraction,
                num_func_samples=32,
                resample=True,
                num_iters_wasserstein=list_num_wasserstein_iterations[step],
                hiden_layer_size_lipschitz_nn=512,
                num_iters_lipschitz=5,
                lipschitz_func_pretraining=False,
                output_subdirectory=output_subdirectory_parameters,
                project_directory=project_directory,
                device=device,
            )

        num_parameters = model.num_parameters
        parameter_names = model.parameter_names

        gaussian_process = create_gp()
        select_gp_prior()
        infer_gp_posterior()
        parameter_distribution = extract_parameter_distribution()

        parameter_moments, parameter_samples = sample_and_analyse_distribution(
            parameter_distribution, num_samples_parameter_distribution
        )

        plot_histograms(
            parameter_names=parameter_names,
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=parameter_moments,
            samples=parameter_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_parameters,
            project_directory=project_directory,
        )
        plot_model_stresses(
            model=model,
            model_parameter_samples=parameter_samples,
            subdirectory_name="model_full",
            output_directory=output_directory_step,
        )

        if is_first_step:
            select_model_through_sobol_sensitivity_analysis(
                model=model,
                parameter_distribution=parameter_distribution,
                first_sobol_index_thresshold=first_sobol_index_thresshold,
                num_samples_factor=num_samples_factor_sensitivity_analysis,
                data_set_label=data_set_label,
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_subdirectory_sensitivities,
                project_directory=project_directory,
                device=device,
            )
            plot_model_stresses(
                model=model,
                model_parameter_samples=parameter_samples,
                subdirectory_name="model_selected",
                output_directory=output_directory_step,
            )
            model.reduce_to_activated_parameters()

        save_model_state(model, output_directory_step, project_directory)
else:
    for step in range(num_discovery_steps):
        is_first_step = step == 0
        output_directory_step = os.path.join(output_directory, f"discovery_step_{step}")
        output_subdirectory_parameters = os.path.join(
            output_directory_step, output_subdirectory_name_parameters
        )
        output_subdirectory_sensitivities = os.path.join(
            output_directory_step, output_subdirectory_name_sensitivities
        )
        if is_first_step:
            input_directory_step = output_directory_step
        else:
            input_step = 0
            input_directory_step = os.path.join(
                output_directory, f"discovery_step_{input_step}"
            )

        if not is_first_step:
            load_model_state(model, input_directory_step, project_directory, device)

        num_parameters = model.num_parameters
        parameter_names = model.parameter_names

        def load_parameter_distribution() -> DistributionProtocol:
            return load_normalizing_flow_parameter_distribution(
                model=model,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
                device=device,
            )

        parameter_distribution = load_parameter_distribution()
        parameter_moments, parameter_samples = sample_and_analyse_distribution(
            parameter_distribution, num_samples_parameter_distribution
        )

        plot_histograms(
            parameter_names=parameter_names,
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=parameter_moments,
            samples=parameter_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_parameters,
            project_directory=project_directory,
        )
        plot_model_stresses(
            model=model,
            model_parameter_samples=parameter_samples,
            subdirectory_name="model_full",
            output_directory=output_directory_step,
        )

        if is_first_step:
            select_model_through_sobol_sensitivity_analysis(
                model=model,
                parameter_distribution=parameter_distribution,
                first_sobol_index_thresshold=first_sobol_index_thresshold,
                num_samples_factor=num_samples_factor_sensitivity_analysis,
                data_set_label=data_set_label,
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_subdirectory_sensitivities,
                project_directory=project_directory,
                device=device,
            )
            plot_model_stresses(
                model=model,
                model_parameter_samples=parameter_samples,
                subdirectory_name="model_selected",
                output_directory=output_directory_step,
            )
