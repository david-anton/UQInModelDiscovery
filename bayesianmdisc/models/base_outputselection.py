from typing import Protocol

from bayesianmdisc.customtypes import TensorSize
from bayesianmdisc.errors import OutputSelectorError
from bayesianmdisc.models.base import StressOutputs


class OutputSelectorProtocol(Protocol):

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
