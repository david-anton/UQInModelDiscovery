import torch

from uqmodeldisc.customtypes import Tensor


class TanhConstrainedFlow(torch.nn.Module):
    def __init__(
        self,
        num_outputs: int,
        indices_constrained_outputs: list[int],
        lower_limit_scale: Tensor,
        higher_limit_scale: Tensor,
    ) -> None:
        super().__init__()
        self._num_outputs = num_outputs
        self._indices_constr = indices_constrained_outputs
        self._indices_unconstr = self._find_indices_of_unconstrained_outputs()
        self._permutation, self._inv_permutation = self._set_up_permutations()
        self._lower_scale = torch.absolute(lower_limit_scale)
        self._higher_scale = torch.absolute(higher_limit_scale)

    def forward(self, u) -> tuple[Tensor, Tensor]:
        constr_u = self._unsqueeze_if_necessary(u[:, self._indices_constr])
        mask_neg_constr_u = constr_u < 0.0
        unconstr_u = self._unsqueeze_if_necessary(u[:, self._indices_unconstr])

        tanh_constr_u = torch.tanh(constr_u)

        def x_func() -> Tensor:
            constr_x = torch.where(
                mask_neg_constr_u,
                tanh_constr_u * self._lower_scale,
                tanh_constr_u * self._higher_scale,
            )
            return torch.concat((constr_x, unconstr_u), dim=1)[:, self._inv_permutation]

        def log_det_func() -> Tensor:
            d_tanh_constr_u_du = 1.0 - torch.square(tanh_constr_u)
            d_constr_x_du = torch.where(
                mask_neg_constr_u,
                d_tanh_constr_u_du * self._lower_scale,
                d_tanh_constr_u_du * self._higher_scale,
            )
            return torch.sum(torch.log(torch.absolute(d_constr_x_du)), dim=1)

        x = x_func()
        log_det = log_det_func()
        return x, log_det

    def inverse(self, x) -> tuple[Tensor, Tensor]:
        constr_x = self._unsqueeze_if_necessary(x[:, self._indices_constr])
        mask_neg_constr_x = constr_x < 0.0
        unconstr_x = self._unsqueeze_if_necessary(x[:, self._indices_unconstr])

        def u_func() -> Tensor:
            constr_u = torch.where(
                mask_neg_constr_x,
                torch.atanh(x / self._lower_scale),
                torch.atanh(x / self._higher_scale),
            )
            return torch.concat((constr_u, unconstr_x), dim=1)[:, self._inv_permutation]

        def log_det_func() -> Tensor:
            d_constr_u_dx = torch.where(
                mask_neg_constr_x,
                self._lower_scale / (self._lower_scale**2 - constr_x**2),
                self._higher_scale / (self._higher_scale**2 - constr_x**2),
            )
            return torch.sum(torch.log(torch.absolute(d_constr_u_dx)), dim=1)

        u = u_func()
        log_det = log_det_func()
        return u, log_det

    def _find_indices_of_unconstrained_outputs(self) -> list[int]:
        return [
            index
            for index in range(self._num_outputs)
            if index not in self._indices_constr
        ]

    def _set_up_permutations(self) -> tuple[list[int], list[int]]:
        permutation = self._indices_constr + self._indices_unconstr
        inverse_permutation = [0 for i in range(self._num_outputs)]
        for i in range(self._num_outputs):
            inverse_permutation[permutation[i]] = i

        return permutation, inverse_permutation

    def _unsqueeze_if_necessary(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.unsqueeze(x, dim=1)
        return x


def create_tanh_constrained_flow(
    total_num_outputs: int,
    indices_constrained_outputs: list[int],
    lower_limit_scale: Tensor,
    higher_limit_scale: Tensor,
) -> TanhConstrainedFlow:
    return TanhConstrainedFlow(
        total_num_outputs,
        indices_constrained_outputs,
        lower_limit_scale,
        higher_limit_scale,
    )
