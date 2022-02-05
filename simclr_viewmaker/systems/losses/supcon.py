from itertools import combinations
import torch
from torch import nn, Tensor
from typing import Tuple

class SupConLoss(nn.Module):
  def __init__(self, temp: float) -> Tensor:
    super().__init__()
    self.temp = temp
  
  def forward(self, zs: Tuple[Tensor, ...], lbls: Tensor) -> Tensor:
    lbls = torch.cat((lbls, lbls), dim=0)
    diff = (lbls.view(-1, 1) != lbls).fill_diagonal_(True)
    same_sum = (~diff).sum(dim=-1)
    not_diag = ~torch.eye(len(lbls), dtype=torch.bool, device=lbls.device)

    losses = []
    for z1, z2 in combinations(zs, 2):
      z = torch.cat((z1, z2), dim=0)

      exp_sim = torch.exp(torch.mm(z, z.T) / self.temp)
      denom = exp_sim.masked_select(not_diag).view(len(z), -1).sum(dim=-1, keepdim=True)

      #masked_fill ensures losses only come from pairs with same label
      loss = -torch.log(exp_sim / denom).masked_fill_(diff, 0)

      losses.append((loss.sum(dim=-1) / same_sum).mean())

    return torch.stack(losses).mean()