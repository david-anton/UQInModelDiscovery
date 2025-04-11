import pandas as pd
import torch

from bayesianmdisc.customtypes import Device
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader, PandasDataWriter
from bayesianmdisc.models import ModelProtocol

file_name_parameter_population_indices = "parameter_population_indices"


def save_model_state(
    model: ModelProtocol, output_directory: str, project_directory: ProjectDirectory
) -> None:
    data_writer = PandasDataWriter(project_directory)
    pupulation_indices = model.get_model_state()
    data_writer.write(
        data=pd.DataFrame(pupulation_indices.detach().cpu().numpy()),
        file_name=file_name_parameter_population_indices,
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
    population_indices = data_reader.read(
        file_name_parameter_population_indices,
        output_directory,
        header=False,
        read_from_output_dir=True,
    )
    model.init_model_state(
        parameter_population_indices=torch.from_numpy(population_indices)
        .type(torch.get_default_dtype())
        .to(device)
    )
