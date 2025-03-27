from typing import Protocol, TypeAlias

from bayesianmdisc.types import Tensor

Inputs: TypeAlias = Tensor
Outputs: TypeAlias = Tensor
DeformationGradient: TypeAlias = Tensor
Stretch: TypeAlias = Tensor
Stretches: TypeAlias = Tensor
HydrostaticPressure: TypeAlias = Tensor
Invariant: TypeAlias = Tensor
Invariants: TypeAlias = tuple[Invariant, ...]
CauchyStress: TypeAlias = Tensor
CauchyStresses: TypeAlias = Tensor
CauchyStressTensor: TypeAlias = Tensor
CauchyStressTensors: TypeAlias = Tensor
StrainEnergy: TypeAlias = Tensor
StrainEnergyDerivative: TypeAlias = Tensor
StrainEnergyDerivatives: TypeAlias = tuple[StrainEnergyDerivative, ...]
Parameters: TypeAlias = Tensor
SplittedParameters: TypeAlias = tuple[Parameters, ...]


class ModelLibrary(Protocol):
    output_dim: int
    num_parameters: int

    def __call__(self, inputs: Inputs, parameters: Parameters) -> CauchyStressTensors:
        pass

    def forward(self, inputs: Inputs, parameters: Parameters) -> CauchyStressTensors:
        pass
