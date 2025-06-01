from typing import TypeAlias

import normflows as nf
import torch

MaskedAutoregressiveFlow: TypeAlias = nf.flows.MaskedAffineAutoregressive


def create_masked_autoregressive_flow(
    number_inputs: int, width_hidden_layer: int
) -> MaskedAutoregressiveFlow:
    return nf.flows.MaskedAffineAutoregressive(
        features=number_inputs,
        hidden_features=width_hidden_layer,
        context_features=None,
        num_blocks=1,
        use_residual_blocks=False,
        random_mask=False,
        activation=torch.nn.functional.leaky_relu,  # torch.nn.functional.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    )
