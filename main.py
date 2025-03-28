import os
from datetime import date

import torch

from bayesianmdisc.bayes.prior import PriorProtocol
from bayesianmdisc.data import LinkaHeartDataReader
from bayesianmdisc.errors import DataError
from bayesianmdisc.gppriors import infer_gp_induced_prior
from bayesianmdisc.gps import (
    IndependentMultiOutputGP,
    create_scaled_rbf_gaussian_process,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import LinkaCANN
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.types import Tensor

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


def create_gp_prior(inputs: Tensor, outputs: Tensor) -> PriorProtocol:

    def validate_number_of_samples(inputs: Tensor, outputs: Tensor) -> None:
        num_inputs = len(inputs)
        num_outputs = len(outputs)

        if num_inputs != num_outputs:
            raise DataError(
                f"""The number of inputs and outputs is expected to be the same,
                but is {num_inputs} and {num_outputs}"""
            )
        else:
            return num_inputs

    def create_gaussian_process() -> IndependentMultiOutputGP:
        jitter = 1e-7
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

        return IndependentMultiOutputGP(gps=tuple(gaussian_processes), device=device)

    validate_number_of_samples(inputs, outputs)
    output_subdirectory = os.path.join(output_directory, "prior")
    model_inputs = inputs
    gp_inputs = model_inputs[:, :num_deformation_inputs]

    min_inputs = torch.amin(gp_inputs, dim=0)
    max_inputs = torch.amax(gp_inputs, dim=0)
    input_dim = gp_inputs.size()[1]
    output_dim = outputs.size()[1]
    initial_noise_stddev = 1e-3

    gaussian_process = create_gaussian_process()

    optimize_gp_hyperparameters(
        gaussian_process=gaussian_process,
        inputs=gp_inputs,
        outputs=outputs,
        initial_noise_standard_deviations=torch.tensor(
            [initial_noise_stddev for _ in range(output_dim)], device=device
        ),
        num_iterations=int(5e4),
        learning_rate=5e-3,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    prior = infer_gp_induced_prior(
        gp=gaussian_process,
        model=model,
        prior_type="Gamma",
        is_mean_trainable=True,
        inputs=model_inputs,
        num_deformation_inputs=num_deformation_inputs,
        num_func_samples=64,
        resample=True,
        num_iters_wasserstein=int(1e4),
        hiden_layer_size_lipschitz_nn=256,
        num_iters_lipschitz=10,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )
    return prior


prior = create_gp_prior(inputs, outputs)
