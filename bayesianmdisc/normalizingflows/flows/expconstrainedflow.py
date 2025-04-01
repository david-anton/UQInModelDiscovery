import torch

from bayesianmdisc.customtypes import Tensor


class ExpConstrainedFlow(torch.nn.Module):
    def __init__(
        self,
        num_outputs: int,
        indices_constrained_outputs: list[int],
    ) -> None:
        super().__init__()
        self._num_outputs = num_outputs
        self._indices_constr = indices_constrained_outputs
        self._indices_unconstr = self._find_indices_of_unconstrained_outputs()
        self._permutation, self._inv_permutation = self._set_up_permutations()

    def forward(self, u) -> tuple[Tensor, Tensor]:
        constr_u = self._unsqueeze_if_necessary(u[:, self._indices_constr])
        unconstr_u = self._unsqueeze_if_necessary(u[:, self._indices_unconstr])

        exp_constr_u = torch.exp(constr_u)

        def x_func() -> Tensor:
            constr_x = exp_constr_u
            return torch.concat((constr_x, unconstr_u), dim=1)[:, self._inv_permutation]

        def log_det_func() -> Tensor:
            d_constr_x_du = exp_constr_u
            return torch.sum(torch.log(torch.absolute(d_constr_x_du)), dim=1)

        x = x_func()
        log_det = log_det_func()
        return x, log_det

    def inverse(self, x) -> tuple[Tensor, Tensor]:
        constr_x = self._unsqueeze_if_necessary(x[:, self._indices_constr])
        unconstr_x = self._unsqueeze_if_necessary(x[:, self._indices_unconstr])

        def u_func() -> Tensor:
            constr_u = torch.log(constr_x)
            return torch.concat((constr_u, unconstr_x), dim=1)[:, self._inv_permutation]

        def log_det_func() -> Tensor:
            d_constr_u_dx = 1 / constr_x
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


def create_exponential_constrained_flow(
    total_num_outputs: int,
    indices_constrained_outputs: list[int],
) -> ExpConstrainedFlow:
    return ExpConstrainedFlow(total_num_outputs, indices_constrained_outputs)
