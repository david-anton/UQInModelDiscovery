import os
from datetime import date
from typing import cast

import torch

from bayesianmdisc.bayes.distributions import (
    DistributionProtocol,
    sample_and_analyse_distribution,
    create_univariate_inverse_gamma_distribution,
    create_univariate_gamma_distribution,
    multiply_distributions,
)
from bayesianmdisc.normalizingflows import (
    FitNormalizingFlowConfig,
    LoadNormalizingFlowConfig,
    NormalizingFlowDistribution,
    fit_normalizing_flow,
    load_normalizing_flow,
)
from bayesianmdisc.statistics.utility import (
    determine_moments_of_multivariate_normal_distribution,
    MomentsMultivariateNormal,
)
from bayesianmdisc.bayes.likelihood import create_likelihood, LikelihoodProtocol
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
    add_noise_to_data,
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
    save_normalizing_flow_parameter_distribution,
)
from bayesianmdisc.postprocessing.plot import (
    plot_gp_stresses_treloar,
    plot_gp_stresses_linka,
    plot_histograms,
    plot_model_stresses_kawabata,
    plot_model_stresses_linka,
    plot_model_stresses_treloar,
    plot_sobol_indice_paths_treloar,
    plot_sobol_indice_statistics,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed

data_set_label = data_set_label_treloar
retrain_models = False  # True

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Set up
current_date = date.today().strftime("%Y%m%d")
if data_set_label == data_set_label_treloar:
    input_directory = data_set_label
    data_set: DataSetProtocol = TreloarDataSet(
        input_directory, project_directory, device
    )
    model: ModelProtocol = IsotropicModelLibrary(output_dim=1, device=device)
    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 5e-2
    list_num_wasserstein_iterations = [50_000, 50_000]
    first_sobol_index_thresshold = 1e-6  # 1e-5
elif data_set_label == data_set_label_kawabata:
    input_directory = data_set_label
    data_set = KawabataDataSet(input_directory, project_directory, device)
    model = IsotropicModelLibrary(output_dim=2, device=device)
    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 5e-2
    list_num_wasserstein_iterations = [50_000, 50_000]
    first_sobol_index_thresshold = 1e-5
elif data_set_label == data_set_label_linka:
    input_directory = "heart_data_linka"
    data_set = LinkaHeartDataSet(input_directory, project_directory, device)
    model = OrthotropicCANN(device)
    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 5e-2
    list_num_wasserstein_iterations = [20_000, 20_000]
    first_sobol_index_thresshold = 1e-4

num_samples_parameter_distribution = 8192
num_samples_factor_sensitivity_analysis = 4096


# output_directory = f"{current_date}_{input_directory}_normalizingflow_relnoise5e-2_minabsnoise5e-2_lipschitz_iters10_lambda10_lr1_samples32_layer2_width1024_numinputs8"
output_directory = f"20250529_treloar_test_clamped_leakyrelu_lbfgs"
output_subdirectory_name_gp = "gp"
output_subdirectory_name_parameters = "parameters"
output_subdirectory_name_sensitivities = "sensitivity_analysis"


def plot_gp_stresses(
    gaussian_process: GaussianProcess,
    data_set_label: str,
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

    def plot_linka() -> None:
        plot_gp_stresses_linka(
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
    elif data_set_label == data_set_label_linka:
        plot_linka()


def plot_model_stresses(
    model: ModelProtocol,
    model_parameter_samples: NPArray,
    data_set_label: str,
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
            test_cases=test_cases.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
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


def plot_relevenat_sobol_indices_results(
    relevant_parameter_indices: list[int],
    data_set_label: str,
    num_outputs: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    plot_sobol_indice_statistics(
        relevant_parameter_indices=relevant_parameter_indices,
        num_outputs=num_outputs,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
    )
    if data_set_label == data_set_label_treloar:
        plot_sobol_indice_paths_treloar(
            relevant_parameter_indices=relevant_parameter_indices,
            inputs=inputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )


def perform_baysian_inference_on_kawabata_data(
    model: IsotropicModelLibrary,
    parameter_distribution: DistributionProtocol,
    output_directory_step: str,
) -> None:
    def reduce_to_model_posterior(
        posterior_samples: NPArray,
        posterior_moments: MomentsMultivariateNormal,
    ) -> tuple[NPArray, MomentsMultivariateNormal]:
        if estimate_noise:
            model_posterior_samples = posterior_samples[:, 1:]
            model_posterior_moments = (
                determine_moments_of_multivariate_normal_distribution(
                    model_posterior_samples
                )
            )
        else:
            model_posterior_samples = posterior_samples
            model_posterior_moments = posterior_moments
        return model_posterior_samples, model_posterior_moments

    def determine_number_of_parameters() -> int:
        num_parameters = model.num_parameters
        if estimate_noise:
            num_parameters += num_noise_parameters
        return num_parameters

    def determine_parameter_names() -> tuple[str, ...]:
        parameter_names = model.parameter_names
        if estimate_noise:
            parameter_names = noise_parameter_name + parameter_names
        return parameter_names

    # print("#########################################################")
    # print("#########################################################")
    # mr_mr = 1.81 * 1e-4
    # mr_nh = 1.15 * 1e-1
    # mr_c20 = 4.58 * 1e-7
    # mr_c30 = 2.77 * 1e-5
    # o_m05 = 3.65 * 1e-1
    # o_m025 = 3.65
    # o_025 = 2.27 * 1e-3
    # parameters_sample = torch.tensor(
    #     [mr_mr, mr_nh, mr_c20, mr_c30, o_m05, o_m025, o_025]
    # )
    # print(parameter_distribution.log_prob(parameters_sample))
    # print(parameter_distribution.prob(parameters_sample))
    # print("#########################################################")
    # print("#########################################################")
    input_directory = data_set_label_kawabata
    output_directory_name = "bayesian_inference_kawabata"
    output_subdirectory_name_parameters = "parameters"
    output_directory = os.path.join(output_directory_step, output_directory_name)
    output_subdirectory_parameters = os.path.join(
        output_directory, output_subdirectory_name_parameters
    )

    output_dim = 2
    model.set_output_dimension(output_dim)
    relative_noise_stddevs = None  # 5e-2
    min_absolute_noise_stddev = 1e-2  # 5e-2
    estimate_noise = relative_noise_stddevs == None
    num_noise_parameters = 1
    noise_parameter_name = ("rel. noise standard deviation",)

    data_set_kawabata = KawabataDataSet(input_directory, project_directory, device)
    inputs, test_cases, outputs = data_set_kawabata.read_data()

    if relative_noise_stddevs != None:
        noise_stddevs = determine_heteroscedastic_noise(
            cast(float, relative_noise_stddevs),
            min_absolute_noise_stddev,
            outputs,
        )
        validate_data(inputs, test_cases, outputs, noise_stddevs)

    num_parameters = determine_number_of_parameters()
    parameter_names = determine_parameter_names()

    num_flows = 16
    relative_width_flow_layers = 4
    retrain_models = True
    if retrain_models:

        def _create_likelihood() -> LikelihoodProtocol:
            return create_likelihood(
                model=model,
                relative_noise_stddev=relative_noise_stddevs,
                min_noise_stddev=min_absolute_noise_stddev,
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                device=device,
            )

        def _create_prior() -> DistributionProtocol:
            if estimate_noise:
                # prior_noise = create_univariate_inverse_gamma_distribution(
                #     concentration=2.0,
                #     rate=0.1,
                #     device=device,
                # )
                prior_noise = create_univariate_gamma_distribution(
                    concentration=1.0,
                    rate=10.0,
                    device=device,
                )
                return multiply_distributions([prior_noise, parameter_distribution])
            else:
                return parameter_distribution

        likelihood = _create_likelihood()
        prior = _create_prior()

        fit_normalizing_flow_config = FitNormalizingFlowConfig(
            likelihood=likelihood,
            prior=prior,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            num_samples=64,
            initial_learning_rate=1e-5,
            final_learning_rate=1e-5,
            num_iterations=2000,
            output_subdirectory=output_directory,
            project_directory=project_directory,
        )

        normalizing_flow = fit_normalizing_flow(fit_normalizing_flow_config, device)
    else:
        load_normalizing_flow_config = LoadNormalizingFlowConfig(
            num_parameters=num_parameters,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            output_subdirectory=output_directory,
            project_directory=project_directory,
        )
        normalizing_flow = load_normalizing_flow(load_normalizing_flow_config, device)

    normalizing_flow_distribution = NormalizingFlowDistribution(
        normalizing_flow, device
    )
    posterior_moments, posterior_samples = sample_and_analyse_distribution(
        normalizing_flow_distribution, num_samples_parameter_distribution
    )
    model_posterior_samples, _ = reduce_to_model_posterior(
        posterior_samples, posterior_moments
    )

    plot_histograms(
        parameter_names=parameter_names,
        true_parameters=tuple(None for _ in range(num_parameters)),
        moments=posterior_moments,
        samples=posterior_samples,
        algorithm_name="nf",
        output_subdirectory=output_subdirectory_parameters,
        project_directory=project_directory,
    )
    plot_model_stresses_kawabata(
        model=cast(IsotropicModelLibrary, model),
        parameter_samples=model_posterior_samples,
        inputs=inputs.detach().cpu().numpy(),
        outputs=outputs.detach().cpu().numpy(),
        test_cases=test_cases.detach().cpu().numpy(),
        output_subdirectory=output_directory,
        project_directory=project_directory,
        device=device,
    )


inputs, test_cases, outputs = data_set.read_data()
noise_stddevs = determine_heteroscedastic_noise(
    relative_noise_stddevs, min_absolute_noise_stddev, outputs
)
# if data_set_label == data_set_label_linka:
#     outputs = add_noise_to_data(noise_stddevs, outputs, device)

validate_data(inputs, test_cases, outputs, noise_stddevs)
num_discovery_steps = len(list_num_wasserstein_iterations)

if retrain_models:
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
            if data_set_label == data_set_label_treloar:
                num_iterations = int(2e4)  # int(5e4)
                learning_rate = 5e-3  # 1e-5
            elif data_set_label == data_set_label_linka:
                num_iterations = int(2e4)
                learning_rate = 5e-3

            optimize_gp_hyperparameters(
                gaussian_process=gaussian_process,
                inputs=inputs,
                outputs=outputs,
                initial_noise_stddevs=noise_stddevs,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
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
                data_set_label=data_set_label,
                output_subdirectory=output_subdirectory_gp,
            )

        def extract_parameter_distribution() -> DistributionProtocol:
            if data_set_label == data_set_label_treloar:
                data_set_treloar = cast(TreloarDataSet, data_set)
                inputs_extraction, test_cases_extraction = (
                    data_set_treloar.generate_uniform_inputs(
                        num_points_per_test_case=32
                    )
                )
                num_func_samples = 32
                hiden_layer_size_lipschitz_nn = 512
                num_iters_lipschitz = 10
            elif data_set_label == data_set_label_linka:
                data_set_linka = cast(LinkaHeartDataSet, data_set)
                inputs_extraction, test_cases_extraction = (
                    data_set_linka.generate_uniform_inputs(num_points_per_test_case=8)
                )
                num_func_samples = 32
                hiden_layer_size_lipschitz_nn = 1024
                num_iters_lipschitz = 10
            distribution = extract_gp_inducing_parameter_distribution(
                gp=gaussian_process,
                model=model,
                distribution_type="normalizing flow",
                is_mean_trainable=True,
                inputs=inputs_extraction,
                test_cases=test_cases_extraction,
                num_func_samples=num_func_samples,
                lipschitz_penalty_coefficient=10.0,
                resample=True,
                num_iters_wasserstein=list_num_wasserstein_iterations[step],
                hiden_layer_size_lipschitz_nn=hiden_layer_size_lipschitz_nn,
                num_iters_lipschitz=num_iters_lipschitz,
                lipschitz_func_pretraining=False,
                output_subdirectory=output_subdirectory_parameters,
                project_directory=project_directory,
                device=device,
            )
            if isinstance(distribution, NormalizingFlowDistribution):
                save_normalizing_flow_parameter_distribution(
                    distribution, output_directory_step, project_directory, device
                )
            return distribution

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
            data_set_label=data_set_label,
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
                output_subdirectory=output_subdirectory_sensitivities,
                project_directory=project_directory,
                device=device,
            )
            plot_relevenat_sobol_indices_results(
                relevant_parameter_indices=model.get_active_parameter_indices(),
                data_set_label=data_set_label,
                num_outputs=model.output_dim,
                output_subdirectory=output_subdirectory_sensitivities,
                project_directory=project_directory,
            )
            plot_model_stresses(
                model=model,
                model_parameter_samples=parameter_samples,
                data_set_label=data_set_label,
                subdirectory_name="model_selected",
                output_directory=output_directory_step,
            )
            model.reduce_to_activated_parameters()

        if not is_first_step and data_set_label == data_set_label_treloar:
            perform_baysian_inference_on_kawabata_data(
                model=cast(IsotropicModelLibrary, model),
                parameter_distribution=parameter_distribution,
                output_directory_step=output_directory_step,
            )

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

        # plot_histograms(
        #     parameter_names=parameter_names,
        #     true_parameters=tuple(None for _ in range(num_parameters)),
        #     moments=parameter_moments,
        #     samples=parameter_samples,
        #     algorithm_name="nf",
        #     output_subdirectory=output_subdirectory_parameters,
        #     project_directory=project_directory,
        # )
        # plot_model_stresses(
        #     model=model,
        #     model_parameter_samples=parameter_samples,
        #     data_set_label=data_set_label,
        #     subdirectory_name="model_full",
        #     output_directory=output_directory_step,
        # )

        if is_first_step:
            select_model_through_sobol_sensitivity_analysis(
                model=model,
                parameter_distribution=parameter_distribution,
                first_sobol_index_thresshold=first_sobol_index_thresshold,
                num_samples_factor=num_samples_factor_sensitivity_analysis,
                data_set_label=data_set_label,
                inputs=inputs,
                test_cases=test_cases,
                output_subdirectory=output_subdirectory_sensitivities,
                project_directory=project_directory,
                device=device,
            )
            # plot_relevenat_sobol_indices_results(
            #     relevant_parameter_indices=model.get_active_parameter_indices(),
            #     data_set_label=data_set_label,
            #     num_outputs=model.output_dim,
            #     output_subdirectory=output_subdirectory_sensitivities,
            #     project_directory=project_directory,
            # )
            # plot_model_stresses(
            #     model=model,
            #     model_parameter_samples=parameter_samples,
            #     data_set_label=data_set_label,
            #     subdirectory_name="model_selected",
            #     output_directory=output_directory_step,
            # )

        if not is_first_step and data_set_label == data_set_label_treloar:
            perform_baysian_inference_on_kawabata_data(
                model=cast(IsotropicModelLibrary, model),
                parameter_distribution=parameter_distribution,
                output_directory_step=output_directory_step,
            )


# data_set_label = data_set_label_treloar
# use_gp_prior = True
# retrain_normalizing_flow = True

# # Settings
# settings = Settings()
# project_directory = ProjectDirectory(settings)
# device = get_device()
# set_default_dtype(torch.float64)
# set_seed(0)

# # Input/output
# current_date = date.today().strftime("%Y%m%d")
# if data_set_label == data_set_label_treloar:
#     input_directory = data_set_label
#     data_reader: DataReaderProtocol = TreloarDataReader(
#         input_directory, project_directory, device
#     )
#     model: ModelProtocol = IsotropicModelLibrary(output_dim=1, device=device)
# elif data_set_label == data_set_label_kawabata:
#     input_directory = data_set_label
#     data_reader = KawabataDataReader(input_directory, project_directory, device)
#     model = IsotropicModelLibrary(output_dim=2, device=device)
# elif data_set_label == data_set_label_linka:
#     input_directory = "heart_data_linka"
#     data_reader = LinkaHeartDataReader(input_directory, project_directory, device)
#     model = OrthotropicCANN(device)

# prior_relative_noise_stddevs = 5e-2
# min_noise_stddev = 1e-3
# estimate_noise = False  # True
# num_calibration_steps = 2
# list_num_wasserstein_iterations = [20_000, 10_000]
# list_relative_selection_thressholds = [2.0]
# num_flows = 16
# relative_width_flow_layers = 4
# trim_metric = "mae"
# num_samples_posterior = 4096


# output_directory = (
#     f"{current_date}_{input_directory}_threshold_2_mae_zeromean_conditioned_nfprior"
# )
# output_subdirectory_name_prior = "prior"
# output_subdirectory_name_posterior = "posterior"


# def determine_number_of_model_parameters(model: ModelProtocol) -> int:
#     return model.num_parameters


# def determine_number_of_parameters(model: ModelProtocol) -> int:
#     num_noise_parameters = 1
#     num_parameters = determine_number_of_model_parameters(model)
#     if estimate_noise:
#         num_parameters += num_noise_parameters
#     return num_parameters


# def determine_parameter_names(model: ModelProtocol) -> tuple[str, ...]:
#     noise_parameter_name = ("rel. noise standard deviation",)
#     parameter_names = model.parameter_names
#     if estimate_noise:
#         parameter_names = noise_parameter_name + parameter_names
#     return parameter_names


# def reduce_to_model_posterior(
#     posterior_moments: MomentsMultivariateNormal, posterior_samples: NPArray
# ) -> tuple[MomentsMultivariateNormal, NPArray]:
#     if estimate_noise:
#         model_posterior_samples = posterior_samples[:, 1:]
#         model_posterior_moments = determine_moments_of_multivariate_normal_distribution(
#             model_posterior_samples
#         )
#     else:
#         model_posterior_moments = posterior_moments
#         model_posterior_samples = posterior_samples
#     return model_posterior_moments, model_posterior_samples


# if retrain_normalizing_flow:
#     for step in range(num_calibration_steps):
#         is_last_step = step == num_calibration_steps - 1
#         output_directory_step = os.path.join(
#             output_directory, f"calibration_step_{step}"
#         )
#         num_parameters = determine_number_of_parameters(model)
#         num_model_parameters = determine_number_of_model_parameters(model)
#         parameter_names = determine_parameter_names(model)

#         def _create_likelihood() -> LikelihoodProtocol:
#             if estimate_noise:
#                 relative_noise_stddevs = None
#             else:
#                 relative_noise_stddevs = prior_relative_noise_stddevs
#             return create_likelihood(
#                 model=model,
#                 relative_noise_stddev=relative_noise_stddevs,
#                 min_noise_stddev=min_noise_stddev,
#                 inputs=inputs,
#                 test_cases=test_cases,
#                 outputs=outputs,
#                 device=device,
#             )

#         def _determine_prior() -> PriorProtocol:
#             def _determine_parameter_prior() -> PriorProtocol:
#                 def init_fixed_prior() -> PriorProtocol:
#                     if (
#                         data_set_label == data_set_label_treloar
#                         or data_set_label == data_set_label_kawabata
#                     ):
#                         concentrations = 100.0
#                         rates = 0.01
#                     else:
#                         concentrations = 0.1
#                         rates = 0.1
#                     return (
#                         create_independent_multivariate_inverse_gamma_distributed_prior(
#                             concentrations=torch.tensor(
#                                 [concentrations for _ in range(num_model_parameters)],
#                                 device=device,
#                             ),
#                             rates=torch.tensor(
#                                 [rates for _ in range(num_model_parameters)],
#                                 device=device,
#                             ),
#                             device=device,
#                         )
#                     )

#                 def fit_gp_prior() -> PriorProtocol:
#                     def create_gaussian_process() -> GaussianProcess:
#                         is_single_outut_gp = output_dim == 1
#                         jitter = 1e-7

#                         def create_single_output_gp() -> GP:
#                             gp_mean = "zero"
#                             gaussian_process = create_scaled_rbf_gaussian_process(
#                                 mean=gp_mean,
#                                 input_dim=input_dim,
#                                 min_inputs=min_inputs,
#                                 max_inputs=max_inputs,
#                                 jitter=jitter,
#                                 device=device,
#                             )
#                             initial_parameters_output_scale = [1.0]
#                             initial_parameters_length_scale = [
#                                 0.1 for _ in range(input_dim)
#                             ]
#                             initial_parameters_kernel = (
#                                 initial_parameters_output_scale
#                                 + initial_parameters_length_scale
#                             )

#                             if gp_mean == "linear":
#                                 initial_parameters_weights = [
#                                     1.0 for _ in range(input_dim)
#                                 ]
#                                 initial_parameters_bias = [0.0]
#                                 initial_parameters_mean = (
#                                     initial_parameters_weights + initial_parameters_bias
#                                 )
#                                 initial_parameters = torch.tensor(
#                                     initial_parameters_mean + initial_parameters_kernel,
#                                     device=device,
#                                 )
#                             else:
#                                 initial_parameters = torch.tensor(
#                                     initial_parameters_kernel, device=device
#                                 )
#                             gaussian_process.set_parameters(initial_parameters)
#                             return gaussian_process

#                         def create_multi_output_gp() -> IndependentMultiOutputGP:
#                             gaussian_processes = [
#                                 create_single_output_gp() for _ in range(output_dim)
#                             ]
#                             return IndependentMultiOutputGP(
#                                 gps=tuple(gaussian_processes), device=device
#                             )

#                         if is_single_outut_gp:
#                             return create_single_output_gp()
#                         else:
#                             return create_multi_output_gp()

#                     def determine_prior_moments(
#                         samples: Tensor,
#                     ) -> tuple[MomentsMultivariateNormal, NPArray]:
#                         samples_np = samples.detach().cpu().numpy()
#                         moments = determine_moments_of_multivariate_normal_distribution(
#                             samples_np
#                         )
#                         return moments, samples_np

#                     output_subdirectory = os.path.join(
#                         output_directory_step, output_subdirectory_name_prior
#                     )
#                     min_inputs = torch.amin(inputs, dim=0)
#                     max_inputs = torch.amax(inputs, dim=0)
#                     input_dim = inputs.size()[1]
#                     output_dim = outputs.size()[1]

#                     gaussian_process = create_gaussian_process()

#                     optimize_gp_hyperparameters(
#                         gaussian_process=gaussian_process,
#                         inputs=inputs,
#                         outputs=outputs,
#                         initial_noise_stddevs=noise_stddevs,
#                         num_iterations=int(5e4),
#                         learning_rate=1e-3,
#                         output_subdirectory=output_subdirectory,
#                         project_directory=project_directory,
#                         device=device,
#                     )

#                     condition_gp(
#                         gaussian_process=gaussian_process,
#                         inputs=inputs,
#                         outputs=outputs,
#                         noise_stddevs=noise_stddevs,
#                         device=device,
#                     )

#                     gp_prior = infer_gp_induced_prior(
#                         gp=gaussian_process,
#                         model=model,
#                         prior_type="normalizing flow",
#                         is_mean_trainable=True,
#                         inputs=inputs,
#                         test_cases=test_cases,
#                         num_func_samples=32,
#                         resample=True,
#                         num_iters_wasserstein=list_num_wasserstein_iterations[step],
#                         hiden_layer_size_lipschitz_nn=256,
#                         num_iters_lipschitz=5,
#                         lipschitz_func_pretraining=False,
#                         output_subdirectory=output_subdirectory,
#                         project_directory=project_directory,
#                         device=device,
#                     )

#                     prior_samples = gp_prior.sample(num_samples=4096)
#                     prior_moments, prior_samples_np = determine_prior_moments(
#                         prior_samples
#                     )

#                     plot_histograms(
#                         parameter_names=model.parameter_names,
#                         true_parameters=tuple(None for _ in range(num_parameters)),
#                         moments=prior_moments,
#                         samples=prior_samples_np,
#                         algorithm_name="gp_prior",
#                         output_subdirectory=output_subdirectory,
#                         project_directory=project_directory,
#                     )
#                     return gp_prior

#                 if use_gp_prior:
#                     return fit_gp_prior()
#                 else:
#                     return init_fixed_prior()

#             def _init_relative_noise_stddev_prior() -> PriorProtocol:
#                 return create_univariate_inverse_gamma_distributed_prior(
#                     concentration=2.0,
#                     rate=0.1,
#                     device=device,
#                 )

#             prior_parameters = _determine_parameter_prior()
#             model_prior_samples = prior_parameters.sample(num_samples_posterior)
#             plot_stresses(
#                 model=model,
#                 model_parameter_samples=model_prior_samples.detach().cpu().numpy(),
#                 subdirectory_name="prior_model",
#                 output_directory=output_directory_step,
#             )
#             if estimate_noise:
#                 prior_relative_noise_stddev = _init_relative_noise_stddev_prior()
#                 return multiply_priors([prior_relative_noise_stddev, prior_parameters])
#             else:
#                 return prior_parameters

#         likelihood = _create_likelihood()
#         prior = _determine_prior()

#         fit_normalizing_flow_config = FitNormalizingFlowConfig(
#             likelihood=likelihood,
#             prior=prior,
#             num_flows=num_flows,
#             relative_width_flow_layers=relative_width_flow_layers,
#             num_samples=64,
#             initial_learning_rate=5e-4,
#             final_learning_rate=1e-4,
#             num_iterations=100_000,
#             output_subdirectory=output_directory_step,
#             project_directory=project_directory,
#         )

#         normalizing_flow = fit_normalizing_flow(fit_normalizing_flow_config, device)
#         posterior_moments, posterior_samples = sample_from_normalizing_flow(
#             normalizing_flow, num_samples_posterior
#         )
#         _, model_posterior_samples = reduce_to_model_posterior(
#             posterior_moments, posterior_samples
#         )

#         output_subdirectory_posterior = os.path.join(
#             output_directory_step, output_subdirectory_name_posterior
#         )
#         plot_histograms(
#             parameter_names=parameter_names,
#             true_parameters=tuple(None for _ in range(num_parameters)),
#             moments=posterior_moments,
#             samples=posterior_samples,
#             algorithm_name="nf",
#             output_subdirectory=output_subdirectory_posterior,
#             project_directory=project_directory,
#         )
#         plot_stresses(
#             model=model,
#             model_parameter_samples=model_posterior_samples,
#             subdirectory_name="posterior_model",
#             output_directory=output_directory_step,
#         )

#         if not is_last_step:
#             trim_model(
#                 model=model,
#                 metric=trim_metric,
#                 relative_thresshold=list_relative_selection_thressholds[step],
#                 parameter_samples=torch.from_numpy(model_posterior_samples)
#                 .type(torch.get_default_dtype())
#                 .to(device),
#                 inputs=inputs,
#                 test_cases=test_cases,
#                 outputs=outputs,
#                 output_subdirectory=output_directory_step,
#                 project_directory=project_directory,
#             )
#             plot_stresses(
#                 model=model,
#                 model_parameter_samples=model_posterior_samples,
#                 subdirectory_name="posterior_model_trimmed",
#                 output_directory=output_directory_step,
#             )
#             model.reduce_to_activated_parameters()

#         save_model_state(model, output_directory_step, project_directory)
# else:
#     for step in range(num_calibration_steps):
#         is_first_step = step == 0
#         is_last_step = step == num_calibration_steps - 1
#         output_directory_step = os.path.join(
#             output_directory, f"calibration_step_{step}"
#         )
#         if is_first_step:
#             input_directory_step = output_directory_step
#         else:
#             input_step = step - 1
#             input_directory_step = os.path.join(
#                 output_directory, f"calibration_step_{input_step}"
#             )

#         if not is_first_step:
#             load_model_state(model, input_directory_step, project_directory, device)

#         num_parameters = determine_number_of_parameters(model)
#         parameter_names = determine_parameter_names(model)

#         load_normalizing_flow_config = LoadNormalizingFlowConfig(
#             num_parameters=num_parameters,
#             num_flows=num_flows,
#             relative_width_flow_layers=relative_width_flow_layers,
#             output_subdirectory=output_directory_step,
#             project_directory=project_directory,
#         )
#         normalizing_flow = load_normalizing_flow(load_normalizing_flow_config, device)
#         posterior_moments, posterior_samples = sample_from_normalizing_flow(
#             normalizing_flow, num_samples_posterior
#         )
#         _, model_posterior_samples = reduce_to_model_posterior(
#             posterior_moments, posterior_samples
#         )

#         output_subdirectory_posterior = os.path.join(
#             output_directory_step, output_subdirectory_name_posterior
#         )
#         plot_histograms(
#             parameter_names=parameter_names,
#             true_parameters=tuple(None for _ in range(num_parameters)),
#             moments=posterior_moments,
#             samples=posterior_samples,
#             algorithm_name="nf",
#             output_subdirectory=output_subdirectory_posterior,
#             project_directory=project_directory,
#         )
#         plot_stresses(
#             model=model,
#             model_parameter_samples=model_posterior_samples,
#             subdirectory_name="posterior_model",
#             output_directory=output_directory_step,
#         )

#         if not is_last_step:
#             trim_model(
#                 model=model,
#                 metric=trim_metric,
#                 relative_thresshold=list_relative_selection_thressholds[step],
#                 parameter_samples=torch.from_numpy(model_posterior_samples)
#                 .type(torch.get_default_dtype())
#                 .to(device),
#                 inputs=inputs,
#                 test_cases=test_cases,
#                 outputs=outputs,
#                 output_subdirectory=output_directory_step,
#                 project_directory=project_directory,
#             )
#             plot_stresses(
#                 model=model,
#                 model_parameter_samples=model_posterior_samples,
#                 subdirectory_name="posterior_model_trimmed",
#                 output_directory=output_directory_step,
#             )
