from typing import TypeAlias, Protocol

import numpy as np
import torch

from bayesianmdisc.customtypes import NPArray, Tensor, Device
from bayesianmdisc.data.testcases import TestCases

NPArrayList = list[NPArray]
DeformationInputs: TypeAlias = Tensor
StressOutputs: TypeAlias = Tensor
Data: TypeAlias = tuple[DeformationInputs, TestCases, StressOutputs]

numpy_data_type = np.float64

data_set_label_treloar = "treloar"
data_set_label_kawabata = "kawabata"
data_set_label_linka = "heart_data_linka"


class DataSetProtocol(Protocol):
    def read_data(self) -> Data: ...


def assemble_test_case_identifiers(
    test_case_identifier: int, deformation_input: NPArray
) -> NPArray:
    num_stretches = len(deformation_input)
    return np.full((num_stretches, 1), test_case_identifier, dtype=np.int64)


def convert_to_torch(array: NPArray, device: Device) -> Tensor:
    return torch.from_numpy(array).type(torch.get_default_dtype()).to(device)


def flatten_and_stack_arrays(arrays: list[NPArray]) -> NPArray:
    return stack_arrays([flatten_array(array) for array in arrays])


def flatten_array(array: NPArray) -> NPArray:
    return array.reshape(-1)


def stack_arrays(arrays: list[NPArray]) -> NPArray:
    return np.vstack(arrays)
