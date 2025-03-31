# def _assemble_deformation_gradient(
#     self, stretches: Stretches
# ) -> DeformationGradient:
#     stretch_fiber, stretch_normal = self._split_stretches(stretches)
#     stretch_sheet = self._calculat_stretch_sheet(stretches)
#     row_1 = torch.concat(
#         (
#             torch.unsqueeze(stretch_fiber, dim=0),
#             torch.tensor([0.0], device=self._device),
#             torch.tensor([0.0], device=self._device),
#         )
#     )
#     row_2 = torch.concat(
#         (
#             torch.tensor([0.0], device=self._device),
#             torch.unsqueeze(stretch_sheet, dim=0),
#             torch.tensor([0.0], device=self._device),
#         )
#     )
#     row_3 = torch.concat(
#         (
#             torch.tensor([0.0], device=self._device),
#             torch.tensor([0.0], device=self._device),
#             torch.unsqueeze(stretch_normal, dim=0),
#         )
#     )
#     return torch.stack((row_1, row_2, row_3))
