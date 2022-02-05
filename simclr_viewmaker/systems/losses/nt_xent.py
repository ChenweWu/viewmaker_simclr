from itertools import combinations
import torch
from torch import nn, Tensor
from typing import Tuple

class NTXentLoss(nn.Module):
  def __init__(self, temp: float) -> Tensor:
    super().__init__()
    self.temp = temp
  
  def forward(self, zs: Tuple[Tensor, ...], lbls: Tensor) -> Tensor:
    same = torch.eye(len(lbls), dtype=torch.bool, device=lbls.device).repeat(2, 2).fill_diagonal_(False)
    not_diag = ~torch.eye(len(lbls) * 2, dtype=torch.bool, device=lbls.device)

    losses = []
    for z1, z2 in combinations(zs, 2):
      z = torch.cat((z1, z2), dim=0)

      exp_sim = torch.exp(torch.mm(z, z.T) / self.temp)
      
      loss = -torch.log(exp_sim / torch.sum(exp_sim * not_diag, dim=-1, keepdim=True))
      losses.append(torch.sum(loss[same]))
    return torch.mean(torch.stack(losses))