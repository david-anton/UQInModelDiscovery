import os
from datetime import date

import torch

from bayesianmdisc.bayes.likelihood import Likelihood
from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
)
from bayesianmdisc.data import LinkaHeartDataReader
from bayesianmdisc.errors import DataError
from bayesianmdisc.gppriors import infer_gp_induced_prior
from bayesianmdisc.gps import (
    IndependentMultiOutputGP,
    condition_gp,
    create_scaled_rbf_gaussian_process,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import LinkaCANN
from bayesianmdisc.normalizingflows import NormalizingFlowConfig, fit_normalizing_flow
from bayesianmdisc.postprocessing.plot import plot_posterior_histograms
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.types import Tensor

from bayesianmdisc.postprocessing.plot import plot_stresses_linka_cann

# Input/output
input_directory = "heart_data_linka"
input_file_name = "CANNsHEARTdata_shear05.xlsx"
current_date = date.today().strftime("%Y%m%d")
output_directory = current_date + "_" + input_directory

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


data_reader = LinkaHeartDataReader(
    input_file_name, input_directory, project_directory, device
)
inputs, outputs = data_reader.read()
num_deformation_inputs = data_reader.num_deformation_inputs

model = LinkaCANN(device)
num_parameters = model.num_parameters


def determine_prior_and_noise(
    inputs: Tensor, outputs: Tensor
) -> tuple[PriorProtocol, Tensor]:

    # def validate_number_of_samples(inputs: Tensor, outputs: Tensor) -> None:
    #     num_inputs = len(inputs)
    #     num_outputs = len(outputs)

    #     if num_inputs != num_outputs:
    #         raise DataError(
    #             f"""The number of inputs and outputs is expected to be the same,
    #             but is {num_inputs} and {num_outputs}"""
    #         )
    #     else:
    #         return num_inputs

    # def create_gaussian_process() -> IndependentMultiOutputGP:
    #     jitter = 1e-7
    #     gaussian_processes = [
    #         create_scaled_rbf_gaussian_process(
    #             mean="zero",
    #             input_dims=input_dim,
    #             min_inputs=min_inputs,
    #             max_inputs=max_inputs,
    #             jitter=jitter,
    #             device=device,
    #         )
    #         for _ in range(output_dim)
    #     ]

    #     initial_parameters = torch.tensor(
    #         [1.0] + [0.1 for _ in range(input_dim)], device=device
    #     )
    #     for gaussian_process in gaussian_processes:
    #         gaussian_process.set_parameters(initial_parameters)

    #     return IndependentMultiOutputGP(gps=tuple(gaussian_processes), device=device)

    # validate_number_of_samples(inputs, outputs)
    # output_subdirectory = os.path.join(output_directory, "prior")
    # model_inputs = inputs
    # gp_inputs = model_inputs[:, :num_deformation_inputs]

    # min_inputs = torch.amin(gp_inputs, dim=0)
    # max_inputs = torch.amax(gp_inputs, dim=0)
    # input_dim = gp_inputs.size()[1]
    # output_dim = outputs.size()[1]
    # initial_noise_stddev = 1e-2

    # gaussian_process = create_gaussian_process()

    # optimize_gp_hyperparameters(
    #     gaussian_process=gaussian_process,
    #     inputs=gp_inputs,
    #     outputs=outputs,
    #     initial_noise_standard_deviations=torch.tensor(
    #         [initial_noise_stddev for _ in range(output_dim)], device=device
    #     ),
    #     num_iterations=int(1e4),
    #     learning_rate=5e-3,
    #     output_subdirectory=output_subdirectory,
    #     project_directory=project_directory,
    #     device=device,
    # )
    # condition_gp(
    #     gaussian_process=gaussian_process, inputs=inputs, outputs=outputs, device=device
    # )

    # noise_variance = gaussian_process.get_likelihood_noise_variance()
    # noise_stddevs = torch.sqrt(noise_variance).detach()

    # prior = infer_gp_induced_prior(
    #     gp=gaussian_process,
    #     model=model,
    #     prior_type="Gamma",
    #     is_mean_trainable=True,
    #     inputs=model_inputs,
    #     num_deformation_inputs=num_deformation_inputs,
    #     num_func_samples=64,
    #     resample=True,
    #     num_iters_wasserstein=int(1e4),
    #     hiden_layer_size_lipschitz_nn=128,
    #     num_iters_lipschitz=10,
    #     output_subdirectory=output_subdirectory,
    #     project_directory=project_directory,
    #     device=device,
    # )

    prior = create_independent_multivariate_gamma_distributed_prior(
        concentrations=torch.tensor(
            [1.0 for _ in range(num_parameters)], device=device
        ),
        rates=torch.tensor([5.0 for _ in range(num_parameters)], device=device),
        device=device,
    )

    noise_stddevs = torch.tensor([0.025, 0.025])

    return prior, noise_stddevs


prior, noise_stddevs = determine_prior_and_noise(inputs, outputs)


likelihood = Likelihood(
    model=model,
    noise_stddev=noise_stddevs,
    inputs=inputs,
    outputs=outputs,
    device=device,
)

normalizing_flow_config = NormalizingFlowConfig(
    likelihood=likelihood,
    prior=prior,
    num_flows=32,
    relative_width_flow_layers=4,
    num_samples=64,
    learning_rate=5e-4,
    learning_rate_decay_rate=1.0,
    num_iterations=1,  # 10_000,
    output_subdirectory=output_directory,
    project_directory=project_directory,
)

posterior_moments, posterior_samples = fit_normalizing_flow(
    normalizing_flow_config, device
)

mse_statistics = likelihood.mse_loss_statistics(
    torch.from_numpy(posterior_samples).type(torch.get_default_dtype())
)
print(f"Mean mse: {mse_statistics.mean}")
print(f"Stddev mse: {mse_statistics.stddev}")

output_subdirectory_posterior = os.path.join(output_directory, "posterior")
plot_posterior_histograms(
    parameter_names=model.get_parameter_names(),
    true_parameters=tuple(None for _ in range(num_parameters)),
    moments=posterior_moments,
    samples=posterior_samples,
    algorithm_name="normalizing_flow",
    output_subdirectory=output_subdirectory_posterior,
    project_directory=project_directory,
)
plot_stresses_linka_cann(
    model=model,
    parameter_samples=posterior_samples,
    inputs=inputs.numpy(),
    outputs=outputs.numpy(),
    output_subdirectory=output_directory,
    project_directory=project_directory,
    device=device,
)
