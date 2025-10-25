from typing import Callable, TypeAlias

import normflows as nf
import torch

from uqmodeldisc.customtypes import NFFlow, Tensor

NFFlows: TypeAlias = list[NFFlow]
NFFlowFuncs: TypeAlias = (
    torch.nn.ModuleList | list[Callable[[Tensor], tuple[Tensor, Tensor]]]
)
NFCompositeOutput: TypeAlias = tuple[Tensor, Tensor]


class CompositeFlow(nf.flows.Flow):
    def __init__(self, flows: NFFlows):
        super().__init__()
        self.flows = torch.nn.ModuleList(flows)

    def forward(self, inputs: Tensor) -> NFCompositeOutput:
        device = inputs.device
        output = inputs
        sum_log_det = torch.zeros(len(inputs), device=device)

        for sub_flow in self.flows:
            output, log_det = sub_flow(output)
            sum_log_det = sum_log_det + log_det

        return output, sum_log_det

    def inverse(self, inputs: Tensor) -> NFCompositeOutput:
        device = inputs.device
        outputs = inputs
        sum_log_det = torch.zeros(len(inputs), device=device)

        for sub_flow in reversed(self.flows):
            outputs, log_det = sub_flow.inverse(outputs)
            sum_log_det = sum_log_det + log_det

        return outputs, sum_log_det
