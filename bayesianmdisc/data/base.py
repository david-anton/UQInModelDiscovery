from typing import TypeAlias

from bayesianmdisc.types import NPArray, Tensor

NPArrayList = list[NPArray]
DeformationInputs: TypeAlias = Tensor
StressOutputs: TypeAlias = Tensor
