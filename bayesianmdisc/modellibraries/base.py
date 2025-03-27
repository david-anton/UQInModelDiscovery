from typing import TypeAlias, Protocol

from bayesianmdisc.types import Tensor


DeformationGradient: TypeAlias = Tensor
DeformationGradients: TypeAlias = Tensor
HydrostaticPressure: TypeAlias = Tensor
HydrostaticPressures: TypeAlias = Tensor
Invariant: TypeAlias = Tensor
Invariants: TypeAlias = tuple[Invariant, ...]
CauchyStressTensor: TypeAlias = Tensor
CauchyStressTensors: TypeAlias = Tensor
StrainEnergy: TypeAlias = Tensor
Parameters: TypeAlias = Tensor
SplittedParameters: TypeAlias = tuple[Parameters, ...]


class ModelLibrary(Protocol):
    output_dim: int
    num_parameters: int

    def __call__(
        self,
        deformation_gradients: DeformationGradients,
        hydrostatic_pressures: HydrostaticPressures,
        parameters: Parameters,
    ) -> CauchyStressTensors:
        pass

    def forward(
        self,
        deformation_gradients: DeformationGradients,
        hydrostatic_pressures: HydrostaticPressures,
        parameters: Parameters,
    ) -> CauchyStressTensors:
        pass
