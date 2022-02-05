from . import models
from .losses import SupConLoss
from .models import MLP
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from typing import Dict

class SimCLR(pl.LightningModule):
  def __init__(self, encoder: str, num_channels: int, hparams: Dict) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)
    
    self.num_channels = num_channels
    self.encoder = getattr(models, encoder)(self.num_channels)
    self.mlp = MLP(self.encoder.out_dim, out_dim=self.hparams.embedding_dim)
    self.loss_fn = SupConLoss(self.hparams.temp)

  def forward(self, x: Tensor) -> Tensor:
    N, C, L = x.shape
    x = self._forward_impl(x.view(-1, self.num_channels, L)).view(N, C // self.num_channels, -1)
    return F.normalize(torch.mean(x, dim=1), dim=-1)
  
  def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.encoder(x)
    x = self.mlp(x)
    return x

  def _shared_step(self, batch, stage: str) -> Tensor:
    *augs, pid = batch

    loss = self.loss_fn(tuple(self(aug) for aug in augs), pid)
    self.log(f"pt_{stage}_loss", loss)
    return loss
  
  def training_step(self, batch, batch_idx):
    return self._shared_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._shared_step(batch, "test")
  
  def configure_optimizers(self):
    return Adam(self.parameters(), lr=self.hparams.lr)
  
  def enable_downstream(self):
    self.mlp = nn.Identity()
    del self.loss_fn
    return self
  
  @property
  def out_dim(self):
    return self.hparams.embedding_dim if hasattr(self, "loss_fn") else self.encoder.out_dim