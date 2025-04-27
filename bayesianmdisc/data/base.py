from typing import TypeAlias

import numpy as np

from bayesianmdisc.customtypes import NPArray, Tensor

NPArrayList = list[NPArray]
DeformationInputs: TypeAlias = Tensor
StressOutputs: TypeAlias = Tensor

numpy_data_type = np.float64

data_set_label_treloar = "treloar"
data_set_label_kawabata = "kawabata"
data_set_label_linka = "heart_data_linka"
