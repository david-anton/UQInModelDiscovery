from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from bayesianmdisc.customtypes import Module, Tensor

InitializationFunc: TypeAlias = Callable[[Tensor], Tensor]
Layers: TypeAlias = list[Module]


class LinearHiddenLayer(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        activation: Module,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
        use_layer_norm=False,
    ) -> None:
        super().__init__()
        self._linear_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        self._activation = activation
        self._use_layer_norm = use_layer_norm
        if self._use_layer_norm:
            self._layer_norm = nn.RMSNorm(size_input)
        init_weights(self._linear_layer.weight)
        init_bias(self._linear_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self._use_layer_norm:
            return self._activation(self._linear_layer(self._layer_norm(x)))
        else:
            return self._activation(self._linear_layer(x))


class LinearOutputLayer(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
    ) -> None:
        super().__init__()
        self._fc_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        init_weights(self._fc_layer.weight)
        init_bias(self._fc_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._fc_layer(x)


class FFNN(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation=nn.Tanh(),
        init_weights=nn.init.xavier_uniform_,
        init_bias=nn.init.zeros_,
        use_layer_norm=False,
    ) -> None:
        super().__init__()
        self._layers = self._set_up_layers(
            layer_sizes, activation, init_weights, init_bias, use_layer_norm
        )
        self._output = nn.Sequential(*self._layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self._output(inputs)

    def _set_up_layers(
        self,
        layer_sizes: list[int],
        activation: Module,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
        use_layer_norm: bool,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            LinearHiddenLayer(
                size_input=layer_sizes[i - 1],
                size_output=layer_sizes[i],
                activation=activation,
                init_weights=init_weights,
                init_bias=init_bias,
                use_layer_norm=use_layer_norm,
            )
            for i in range(1, len(layer_sizes) - 1)
        ]

        layer_out = LinearOutputLayer(
            size_input=layer_sizes[-2],
            size_output=layer_sizes[-1],
            init_weights=init_weights,
            init_bias=init_bias,
        )
        layers.append(layer_out)
        return layers
