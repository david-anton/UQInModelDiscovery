import os
from datetime import date

import torch

from bayesianmdisc.bayes.likelihood import Likelihood
from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
    create_independent_multivariate_inverse_gamma_distributed_prior,
)
from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.data import (
    DataReaderProtocol,
    DeformationInputs,
    LinkaHeartDataReader,
    StressOutputs,
    TestCases,
    TreloarDataReader,
)
from bayesianmdisc.errors import DataError
from bayesianmdisc.gppriors import infer_gp_induced_prior
from bayesianmdisc.gps import (
    GP,
    GaussianProcess,
    IndependentMultiOutputGP,
    create_scaled_rbf_gaussian_process,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import IsotropicModelLibrary, ModelProtocol, OrthotropicCANN
from bayesianmdisc.normalizingflows import NormalizingFlowConfig, fit_normalizing_flow
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
use_gp_prior = False

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Input/output
current_date = date.today().strftime("%Y%m%d")
if data_set == "linka":
    input_directory = "heart_data_linka"
    data_reader: DataReaderProtocol = LinkaHeartDataReader(
        input_directory, project_directory, device
    )
elif data_set == "treloar":
    input_directory = "treloar"
    data_reader = TreloarDataReader(input_directory, project_directory, device)
output_directory = current_date + "_" + input_directory  # + "ogden_only"


inputs, test_cases, outputs = data_reader.read()


if data_set == "linka":
    model: ModelProtocol = OrthotropicCANN(device)
elif data_set == "treloar":
    model = IsotropicModelLibrary(device)
num_parameters = model.num_parameters

relative_noise_stddevs = 1e-2
min_noise_stddev = 1e-3


def determine_prior(
    inputs: DeformationInputs, test_cases: TestCases, outputs: StressOutputs
) -> PriorProtocol:

    def validate_number_of_samples(
        inputs: DeformationInputs, test_cases: TestCases, outputs: StressOutputs
    ) -> None:
        num_inputs = len(inputs)
        num_test_cases = len(test_cases)
        num_outputs = len(outputs)

        if (
            num_inputs != num_test_cases
            or num_inputs != num_outputs
            or num_test_cases != num_outputs
        ):
            raise DataError(
                f"""The number of inputs, test cases and outputs is expected to be the same,
                but is {num_inputs}, {num_test_cases} and {num_outputs}"""
            )

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

    def determine_prior_moments(
        samples: Tensor,
    ) -> tuple[MomentsMultivariateNormal, NPArray]:
        samples_np = samples.detach().cpu().numpy()
        moments = determine_moments_of_multivariate_normal_distribution(samples_np)
        return moments, samples_np

    validate_number_of_samples(inputs, test_cases, outputs)
    output_subdirectory = os.path.join(output_directory, "prior")

    if use_gp_prior:
        min_inputs = torch.amin(inputs, dim=0)
        max_inputs = torch.amax(inputs, dim=0)
        input_dim = inputs.size()[1]
        output_dim = outputs.size()[1]

        gaussian_process = create_gaussian_process()

        heteroscedastic_noise_stddevs = relative_noise_stddevs * outputs
        heteroscedastic_noise_stddevs = torch.where(
            heteroscedastic_noise_stddevs < min_noise_stddev,
            min_noise_stddev,
            heteroscedastic_noise_stddevs,
        )

        optimize_gp_hyperparameters(
            gaussian_process=gaussian_process,
            inputs=inputs,
            outputs=outputs,
            initial_noise_stddevs=heteroscedastic_noise_stddevs,
            num_iterations=int(2e4),
            learning_rate=1e-3,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

        prior = infer_gp_induced_prior(
            gp=gaussian_process,
            model=model,
            prior_type="Gamma",
            is_mean_trainable=True,
            inputs=inputs,
            test_cases=test_cases,
            num_func_samples=32,
            resample=True,
            num_iters_wasserstein=int(20e3),
            hiden_layer_size_lipschitz_nn=512,
            num_iters_lipschitz=5,
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
        # return create_independent_multivariate_gamma_distributed_prior(
        #     concentrations=torch.tensor(
        #         [0.1 for _ in range(num_parameters)], device=device
        #     ),
        #     rates=torch.tensor([2.0 for _ in range(num_parameters)], device=device),
        #     device=device,
        # )
        return create_independent_multivariate_inverse_gamma_distributed_prior(
            concentrations=torch.tensor(
                [99.0 for _ in range(num_parameters)], device=device
            ),
            rates=torch.tensor(
                [1 / 10000.0 for _ in range(num_parameters)], device=device
            ),
            device=device,
        )


prior = determine_prior(inputs, test_cases, outputs)

likelihood = Likelihood(
    model=model,
    relative_noise_stddev=relative_noise_stddevs,
    min_noise_stddev=min_noise_stddev,
    inputs=inputs,
    test_cases=test_cases,
    outputs=outputs,
    device=device,
)

normalizing_flow_config = NormalizingFlowConfig(
    likelihood=likelihood,
    prior=prior,
    num_flows=32,
    relative_width_flow_layers=4,
    num_samples=64,
    initial_learning_rate=5e-4,
    final_learning_rate=1e-4,
    num_iterations=100_000,
    deactivate_parameters=True,
    output_subdirectory=output_directory,
    project_directory=project_directory,
)

posterior_moments, posterior_samples = fit_normalizing_flow(
    normalizing_flow_config, device
)

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
