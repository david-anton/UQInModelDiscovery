import os
from dataclasses import dataclass
from datetime import date

import torch

from bayesianmdisc.bayes.likelihood import Likelihood
from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
)
from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.data import (
    DataReaderProtocol,
    DeformationInputs,
    KawabataDataReader,
    LinkaHeartDataReader,
    StressOutputs,
    TestCases,
    TreloarDataReader,
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.errors import DataError, DataSetError
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
from bayesianmdisc.models import IsotropicModelLibrary, ModelProtocol, OrthotropicCANN
from bayesianmdisc.normalizingflows import (
    NormalizingFlowConfig,
    NormalizingFlowProtocol,
    determine_statistical_moments,
    fit_normalizing_flow,
    load_normalizing_flow,
)
from bayesianmdisc.postprocessing.plot import (
    plot_histograms,
    plot_stresses_linka,
    plot_stresses_treloar,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)

data_set = "treloar"
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
if data_set == "treloar":
    input_directory = data_set
    data_reader: DataReaderProtocol = TreloarDataReader(
        input_directory, project_directory, device
    )
if data_set == "kawabata":
    input_directory = data_set
    data_reader = KawabataDataReader(input_directory, project_directory, device)
elif data_set == "linka":
    input_directory = "heart_data_linka"
    data_reader = LinkaHeartDataReader(input_directory, project_directory, device)

output_directory = current_date + "_" + input_directory + "_splitted_data"


if data_set == "linka":
    model: ModelProtocol = OrthotropicCANN(device)
elif data_set == "treloar":
    model = IsotropicModelLibrary(device)
num_parameters = model.num_parameters

relative_noise_stddevs = 5e-2
min_noise_stddev = 1e-3
num_samples_posterior = 4096


def validate_data(
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> None:
    num_inputs = len(inputs)
    num_test_cases = len(test_cases)
    num_outputs = len(outputs)
    num_noise_stddevs = len(noise_stddevs)

    if (
        num_inputs != num_test_cases
        and num_inputs != num_outputs
        and num_inputs != num_noise_stddevs
    ):
        raise DataError(
            f"""The number of inputs, test cases, outputs and noise standard deviations 
                        is expected to be the same but is {num_inputs}, {num_test_cases}, 
                        {num_outputs} and {num_noise_stddevs}"""
        )


def determine_heteroscedastic_noise() -> Tensor:
    noise_stddevs = relative_noise_stddevs * outputs
    return torch.where(
        noise_stddevs < min_noise_stddev,
        min_noise_stddev,
        noise_stddevs,
    )


@dataclass
class SplittedData:
    inputs_prior: DeformationInputs
    inputs_posterior: DeformationInputs
    test_cases_prior: TestCases
    test_cases_posterior: TestCases
    outputs_prior: StressOutputs
    outputs_posterior: StressOutputs
    noise_stddevs_prior: Tensor
    noise_stddevs_posterior: Tensor


def split_data(
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> SplittedData:

    def split_treloar_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)
        mask_ut = torch.where(test_cases == test_case_identifier_uniaxial_tension)[0]
        num_points_ut = torch.numel(mask_ut)
        mask_ebt = torch.where(test_cases == test_case_identifier_equibiaxial_tension)[
            0
        ]
        num_points_ebt = torch.numel(mask_ebt)

        # Relative indices
        rel_indices_prior_ut = [2, 6, 10, 15, 20]
        rel_indices_prior_ebt = [2, 6, 11]
        rel_indices_prior_ps = [2, 5, 10]
        # Absolute indices
        indices_prior_ut = rel_indices_prior_ut
        start_index = num_points_ut
        indices_prior_ebt = [i + start_index for i in rel_indices_prior_ebt]
        start_index = num_points_ut + num_points_ebt
        indices_prior_ps = [i + start_index for i in rel_indices_prior_ps]
        indices_prior = indices_prior_ut + indices_prior_ebt + indices_prior_ps
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    validate_data(inputs, test_cases, outputs, noise_stddevs)
    if data_set == "treloar":
        return split_treloar_data(inputs, test_cases, outputs, noise_stddevs)
    else:
        raise DataSetError(f"No implementation for the requested data set {data_set}")


inputs, test_cases, outputs = data_reader.read()
noise_stddevs = determine_heteroscedastic_noise()
validate_data(inputs, test_cases, outputs, noise_stddevs)


splitted_data = split_data(inputs, test_cases, outputs, noise_stddevs)
inputs_prior = splitted_data.inputs_prior
inputs_posterior = splitted_data.inputs_posterior
test_cases_prior = splitted_data.test_cases_prior
test_cases_posterior = splitted_data.test_cases_posterior
outputs_prior = splitted_data.outputs_prior
outputs_posterior = splitted_data.outputs_posterior
noise_stddevs_prior = splitted_data.noise_stddevs_prior
noise_stddevs_posterior = splitted_data.noise_stddevs_posterior


def determine_prior() -> PriorProtocol:
    if use_gp_prior:

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
            condition_gp(gaussian_process, inputs, outputs, noise_stddevs, device)

        def determine_prior_moments(
            samples: Tensor,
        ) -> tuple[MomentsMultivariateNormal, NPArray]:
            samples_np = samples.detach().cpu().numpy()
            moments = determine_moments_of_multivariate_normal_distribution(samples_np)
            return moments, samples_np

        output_subdirectory = os.path.join(output_directory, "prior")
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

        condition_gaussian_process(inputs_prior, outputs_prior, noise_stddevs_prior)

        prior = infer_gp_induced_prior(
            gp=gaussian_process,
            model=model,
            prior_type="inverse Gamma",
            is_mean_trainable=True,
            inputs=inputs,
            test_cases=test_cases,
            num_func_samples=32,
            resample=True,
            num_iters_wasserstein=int(2e4),
            hiden_layer_size_lipschitz_nn=256,
            num_iters_lipschitz=5,
            lipschitz_func_pretraining=True,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

        prior_samples = prior.sample(num_samples=4096)
        prior_moments, prior_samples_np = determine_prior_moments(prior_samples)

        plot_histograms(
            parameter_names=model.get_parameter_names(),
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=prior_moments,
            samples=prior_samples_np,
            algorithm_name="gp_prior",
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )

        return prior
    else:
        return create_independent_multivariate_gamma_distributed_prior(
            concentrations=torch.tensor(
                [0.5 for _ in range(num_parameters)], device=device
            ),
            rates=torch.tensor([1.0 for _ in range(num_parameters)], device=device),
            device=device,
        )


def sample_from_posterior(
    normalizing_flow: NormalizingFlowProtocol,
) -> tuple[MomentsMultivariateNormal, NPArray]:

    def draw_samples() -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples_posterior)
        return list(samples)

    samples_list = draw_samples()
    return determine_statistical_moments(samples_list)


prior = determine_prior()

likelihood = Likelihood(
    model=model,
    relative_noise_stddev=relative_noise_stddevs,
    min_noise_stddev=min_noise_stddev,
    inputs=inputs_posterior,
    test_cases=test_cases_posterior,
    outputs=outputs_posterior,
    device=device,
)

normalizing_flow_config = NormalizingFlowConfig(
    likelihood=likelihood,
    prior=prior,
    num_flows=16,
    relative_width_flow_layers=4,
    num_samples=64,
    initial_learning_rate=5e-4,
    final_learning_rate=1e-4,
    num_iterations=100_000,
    deactivate_parameters=False,
    output_subdirectory=output_directory,
    project_directory=project_directory,
)

if retrain_normalizing_flow:
    normalizing_flow = fit_normalizing_flow(normalizing_flow_config, device)
else:
    normalizing_flow = load_normalizing_flow(normalizing_flow_config, device)

posterior_moments, posterior_samples = sample_from_posterior(normalizing_flow)

mse_statistics = likelihood.mse_loss_statistics(
    torch.from_numpy(posterior_samples).type(torch.get_default_dtype()).to(device)
)
print(f"Mean mse: {mse_statistics.mean}")
print(f"Stddev mse: {mse_statistics.stddev}")

output_subdirectory_posterior = os.path.join(output_directory, "posterior")

plot_histograms(
    parameter_names=model.get_parameter_names(),
    true_parameters=tuple(None for _ in range(num_parameters)),
    moments=posterior_moments,
    samples=posterior_samples,
    algorithm_name="nf",
    output_subdirectory=output_subdirectory_posterior,
    project_directory=project_directory,
)

if data_set == "linka":
    plot_stresses_linka(
        model=model,
        parameter_samples=posterior_samples,
        inputs=inputs.detach().cpu().numpy(),
        outputs=outputs.detach().cpu().numpy(),
        test_cases=test_cases.detach().cpu().numpy(),
        output_subdirectory=output_directory,
        project_directory=project_directory,
        device=device,
    )
elif data_set == "treloar":
    plot_stresses_treloar(
        model=model,
        parameter_samples=posterior_samples,
        inputs=inputs.detach().cpu().numpy(),
        outputs=outputs.detach().cpu().numpy(),
        test_cases=test_cases.detach().cpu().numpy(),
        output_subdirectory=output_directory,
        project_directory=project_directory,
        device=device,
    )
