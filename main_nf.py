import os
from datetime import date
from typing import cast

import torch

from bayesianmdisc.bayes.distributions import (
    DistributionProtocol,
    create_independent_multivariate_inverse_gamma_distribution,
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
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import (
    IsotropicModelLibrary,
    ModelProtocol,
    OrthotropicCANN,
    save_model_state,
    select_model,
)
from bayesianmdisc.bayes.likelihood import LikelihoodProtocol, create_likelihood
from bayesianmdisc.postprocessing.plot import (
    plot_histograms,
    plot_model_stresses_kawabata,
    plot_model_stresses_linka,
    plot_model_stresses_treloar,
)
from bayesianmdisc.normalizingflows import (
    FitNormalizingFlowConfig,
    fit_normalizing_flow,
    NormalizingFlowDistribution,
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
num_calibration_steps = 2
selection_metric = "mae"
list_relative_selection_thressholds = [0.1]
num_samples_posterior = 4096


output_directory = f"{current_date}_{input_directory}_normalizingflow_relnoise5e-2_minabsnoise5e-2_bayesianinference"
output_subdirectory_name_parameters = "parameters"


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


if retrain_posterior:
    for step in range(num_calibration_steps):
        is_last_step = step == num_calibration_steps - 1
        output_directory_step = os.path.join(
            output_directory, f"calibration_step_{step}"
        )
        output_subdirectory_parameters = os.path.join(
            output_directory_step, output_subdirectory_name_parameters
        )

        def create_prior() -> DistributionProtocol:
            if data_set_label == data_set_label_treloar:
                concentrations = 1.0
                rates = 0.1
            else:
                concentrations = 0.1
                rates = 0.1
            return create_independent_multivariate_inverse_gamma_distribution(
                concentrations=torch.tensor(
                    [concentrations for _ in range(num_parameters)],
                    device=device,
                ),
                rates=torch.tensor(
                    [rates for _ in range(num_parameters)],
                    device=device,
                ),
                device=device,
            )

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

        num_parameters = model.num_parameters
        parameter_names = model.parameter_names

        prior = create_prior()
        likelihood = _create_likelihood()

        fit_normalizing_flow_config = FitNormalizingFlowConfig(
            likelihood=likelihood,
            prior=prior,
            num_flows=16,
            relative_width_flow_layers=4,
            num_samples=32,
            initial_learning_rate=1e-3,
            final_learning_rate=5e-5,
            num_iterations=200_000,
            output_subdirectory=output_directory_step,
            project_directory=project_directory,
        )

        normalizing_flow = fit_normalizing_flow(fit_normalizing_flow_config, device)
        normalizing_flow_distribution = NormalizingFlowDistribution(
            normalizing_flow, device
        )
        parameter_moments, parameter_samples = sample_and_analyse_distribution(
            normalizing_flow_distribution, num_samples_posterior
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

        if not is_last_step:
            select_model(
                model=model,
                metric=selection_metric,
                relative_thresshold=list_relative_selection_thressholds[step],
                parameter_samples=torch.from_numpy(parameter_samples)
                .type(torch.get_default_dtype())
                .to(device),
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
            )
            plot_model_stresses(
                model=model,
                model_parameter_samples=parameter_samples,
                subdirectory_name="model_selected",
                output_directory=output_directory_step,
            )
            model.reduce_to_activated_parameters()

        save_model_state(model, output_directory_step, project_directory)
