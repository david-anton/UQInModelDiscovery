import pandas as pd
import torch

from bayesianmdisc.customtypes import Device, Tensor
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader, PandasDataWriter
from bayesianmdisc.models import ModelProtocol

file_name_parameter_population_indices = "parameter_population_indices"
file_name_parameter_scales = "parameter_scales"


def save_model_state(
    model: ModelProtocol, output_directory: str, project_directory: ProjectDirectory
) -> None:
    data_writer = PandasDataWriter(project_directory)
    pupulation_indices, parameter_scales = model.get_model_state()
    data_writer.write(
        data=pd.DataFrame(pupulation_indices.detach().cpu().numpy()),
        file_name=file_name_parameter_population_indices,
        subdir_name=output_directory,
        header=False,
    )
    data_writer.write(
        data=pd.DataFrame(parameter_scales.detach().cpu().numpy()),
        file_name=file_name_parameter_scales,
        subdir_name=output_directory,
        header=False,
    )


def load_model_state(
    model: ModelProtocol,
    output_directory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    data_reader = CSVDataReader(project_directory)
    population_matrix = data_reader.read_as_numpy_array(
        file_name_parameter_population_indices,
        output_directory,
        header=None,
        read_from_output_dir=True,
    )
    parameter_scales = data_reader.read_as_numpy_array(
        file_name_parameter_scales,
        output_directory,
        header=None,
        read_from_output_dir=True,
    )
    model.init_model_state(
        parameter_population_matrix=torch.from_numpy(population_matrix)
        .type(torch.get_default_dtype())
        .to(device),
        parameter_scales=torch.from_numpy(parameter_scales)
        .type(torch.get_default_dtype())
        .to(device),
    )


def unsqueeze_if_necessary(tensor: Tensor) -> Tensor:
    if tensor.dim() == 0:
        return torch.unsqueeze(tensor, dim=0)
    else:
        return tensor
