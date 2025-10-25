import torch

from uqmodeldisc.customtypes import Device, Tensor


class ScalingFlow(torch.nn.Module):
    def __init__(self, scales: Tensor, device: Device) -> None:
        super().__init__()
        self._scales = scales
        self._device = device

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        num_u = len(u)

        def x_func() -> Tensor:
            return self._scales * u

        def log_det_func() -> Tensor:
            dx_du = self._scales.repeat(num_u, 1)
            return torch.sum(torch.log(torch.absolute(dx_du)), dim=1)

        x = x_func()
        log_det = log_det_func()
        return x, log_det

    def inverse(self, x: Tensor) -> tuple[Tensor, Tensor]:
        num_x = len(x)

        def u_func() -> Tensor:
            return x / self._scales

        def log_det_func() -> Tensor:
            du_dx = 1 / self._scales.repeat(num_x, 1)
            return torch.sum(torch.log(torch.absolute(du_dx)), dim=1)

        u = u_func()
        log_det = log_det_func()
        return u, log_det


def create_scaling_flow(scales: Tensor, device: Device) -> ScalingFlow:
    return ScalingFlow(scales, device)
