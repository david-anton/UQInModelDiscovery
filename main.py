import os
from datetime import date

import torch

from bayesianmdisc.data import LinkaHeartDataReader
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.types import Tensor
from bayesianmdisc.bayes.prior import PriorProtocol
from bayesianmdisc.gps import (
    create_scaled_rbf_gaussian_process,
    IndependentMultiOutputGP,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.errors import DataError

# Input/output
input_directory = "heart_data_linka"
input_file_name = "CANNsHEARTdata_shear05.xlsx"
current_date = date.today().strftime("%Y%m%d")
output_directory = os.path.join(current_date, "_", input_directory)

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Simulation
noise_stddev = 1e-2


data_reader = LinkaHeartDataReader(input_file_name, input_directory, project_directory)
deformation_gradients, stress_tensors = data_reader.read()


# def create_gp_prior(stretches: Tensor, stresses: Tensor) -> PriorProtocol:
#     output_subdirectory = os.path.join(output_directory, "prior")
#     inputs = stretches
#     outputs = stresses

#     def validate_number_of_samples(inputs: Tensor, outputs: Tensor) -> int:
#         num_inputs = len(inputs)
#         num_outputs = len(outputs)

#         if num_inputs != num_outputs:
#             raise DataError(
#                 f"""The number of inputs and outputs is expected to be the same,
#                             but is {num_inputs} and {num_outputs}"""
#             )
#         else:
#             return num_inputs

#     def create_gaussian_process() -> IndependentMultiOutputGP:
#         jitter = 1e-7
#         gaussian_processes = [
#             create_scaled_rbf_gaussian_process(
#                 mean="zero",
#                 input_dims=input_dim,
#                 min_inputs=min_inputs,
#                 max_inputs=max_inputs,
#                 jitter=jitter,
#                 device=device,
#             )
#             for _ in range(output_dim)
#         ]

#         for gaussian_process in gaussian_processes:
#             gaussian_process.set_parameters(
#                 torch.tensor([0.1, 0.1, 0.1], device=device)
#             )

#         return IndependentMultiOutputGP(gps=tuple(gaussian_processes), device=device)

#     num_samples = validate_number_of_samples(inputs, outputs)
#     min_inputs = torch.amin(inputs, dim=0)
#     max_inputs = torch.amax(inputs, dim=0)
#     input_dim = stretches.size()[1]
#     output_dim = stresses.size()[1]

#     gaussian_process = create_gaussian_process()

#     optimize_gp_hyperparameters(
#         gaussian_process=gaussian_process,
#         inputs=inputs,
#         outputs=outputs,
#         initial_noise_standard_deviations=torch.tensor(
#             [noise_stddev for _ in range(output_dim)], device=device
#         ),
#         num_iterations=int(1e4),
#         learning_rate=1e-3,
#         output_subdirectory=output_subdirectory,
#         project_directory=project_directory,
#         device=device,
#     )

#     prior = infer_gp_induced_prior(
#         bayesian_ansatz=ansatz,
#         gp_ansatz=gp_ansatz,
#         prior_type="Gaussian",
#         is_mean_trainable=True,
#         inputs=inputs_gp_inference,
#         num_func_samples=64,
#         resample=True,
#         num_iters_wasserstein=int(10e3),
#         hiden_layer_size_lipschitz_nn=2048,
#         num_iters_lipschitz=5,
#         output_subdirectory=output_subdir_prior_optimization,
#         project_directory=project_directory,
#         device=device,
#     )
#     return prior
