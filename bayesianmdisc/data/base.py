from typing import TypeAlias

import numpy as np

from bayesianmdisc.customtypes import NPArray, Tensor

NPArrayList = list[NPArray]
DeformationInputs: TypeAlias = Tensor
StressOutputs: TypeAlias = Tensor

numpy_data_type = np.float64
