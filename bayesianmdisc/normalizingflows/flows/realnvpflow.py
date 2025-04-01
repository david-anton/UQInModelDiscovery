import math
from typing import TypeAlias

import normflows as nf

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.normalizingflows.flows import CompositeFlow

NFMLP: TypeAlias = nf.nets.MLP


class RealNVPFlow(nf.flows.Flow):
    def __init__(
        self,
        parameter_map: NFMLP,
        scale: bool = True,
        scale_map: str = "exp",
        split_mode: str = "channel",
    ):
        super().__init__()
        flows_ = []
        flows_ += [nf.flows.reshape.Split(split_mode)]
        flows_ += [nf.flows.AffineCoupling(parameter_map, scale, scale_map)]
        flows_ += [nf.flows.reshape.Merge(split_mode)]
        self.flow = CompositeFlow(flows_)

    def forward(self, inputs: Tensor):
        return self.flow.forward(inputs)

    def inverse(self, inputs):
        return self.flow.inverse(inputs)


def create_real_nvp_flow(number_inputs: int, width_hidden_layer: int) -> RealNVPFlow:
    num_parameter_map_inputs = math.ceil(number_inputs / 2)
    num_parameter_map_outputs = 2 * (number_inputs - num_parameter_map_inputs)
    parameter_map = nf.nets.MLP(
        [
            num_parameter_map_inputs,
            width_hidden_layer,
            num_parameter_map_outputs,
        ],
        init_zeros=True,
    )
    flows = [
        RealNVPFlow(
            parameter_map=parameter_map,
            scale=True,
            scale_map="exp",
            split_mode="channel",
        ),
        nf.flows.Permute(num_channels=number_inputs, mode="shuffle"),
    ]
    return CompositeFlow(flows)
