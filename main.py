import os
from datetime import date

import torch

from bayesianmdisc.data import LinkaDataReader
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed

# Input/output
input_directory = "synthetic_data_linka"
current_date = date.today().strftime("%Y%m%d")
output_directory = os.path.join(current_date, "_", input_directory)

# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


data_reader = LinkaDataReader(input_directory, project_directory)
deformation_gradients, stresses, stretches = data_reader.read()


# print(deformation_gradients.shape)
# print(stresses.shape)
# print(stretches.shape)
