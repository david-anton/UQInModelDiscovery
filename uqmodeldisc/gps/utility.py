from typing import TypeAlias, cast

import gpytorch
import torch

from uqmodeldisc.customtypes import Device, GPModel, Module, Tensor, TensorSize
from uqmodeldisc.errors import GPError
from uqmodeldisc.io import ProjectDirectory
from uqmodeldisc.io.loaderssavers import PytorchModelLoader, PytorchModelSaver

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]

file_name_gp_parameters = "gp_parameters"


def validate_parameters_size(
    parameters: Tensor, valid_parameter_size: int | TensorSize
) -> None:
    parameters_size = parameters.size()
    if isinstance(valid_parameter_size, int):
        valid_parameter_size = torch.Size([valid_parameter_size])
    if parameters_size != valid_parameter_size:
        raise GPError(f"Parameter tensor has unvalid size {parameters_size}")


def save_gp(
    gaussian_process: GPModel,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    print("Save GP ...")
    model_saver = PytorchModelSaver(project_directory)
    model_saver.save(
        cast(Module, gaussian_process),
        file_name_gp_parameters,
        output_subdirectory,
        device,
    )


def load_gp(
    gaussian_process: GPModel,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> GPModel:
    print("Load GP ...")
    model_loader = PytorchModelLoader(project_directory)
    return model_loader.load(
        cast(Module, gaussian_process),
        file_name_gp_parameters,
        output_subdirectory,
    ).to(device)
