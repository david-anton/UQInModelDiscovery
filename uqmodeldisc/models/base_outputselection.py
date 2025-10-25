from typing import Protocol

import torch

from uqmodeldisc.customtypes import Tensor, TensorSize
from uqmodeldisc.errors import OutputSelectorError
from uqmodeldisc.models.base import OutputSelectionMask, StressOutputs


class OutputSelectorProtocol(Protocol):
    selection_mask: Tensor
    total_num_selected_outputs: int

    def __call__(self, full_outputs: StressOutputs) -> StressOutputs:
        pass


def validate_full_output_size(
    full_outputs: StressOutputs, expected_size: TensorSize
) -> None:
    full_output_size = full_outputs.size()
    if not full_output_size == expected_size:
        raise OutputSelectorError(
            f"""The full output size {full_output_size} does not match the expected size {expected_size}"""
        )


def determine_full_output_size(
    num_outputs: int, single_full_output_dim: int
) -> TensorSize:
    full_output_dim = int(num_outputs * single_full_output_dim)
    return torch.Size((full_output_dim,))


def count_number_of_selected_outputs(selection_mask: OutputSelectionMask) -> int:
    return int(torch.sum(selection_mask))
