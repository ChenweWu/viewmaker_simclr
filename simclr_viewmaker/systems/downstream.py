from attrdict import AttrDict
from .metrics import AutoThresholdF1
import pytorch_lightning as pl
from torchmetrics import AUROC
import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from torch.optim import Adam
from typing import Dict, Tuple

class DownstreamTask(pl.LightningModule):
  def __init__(self, encoder: nn.Module, prefix: str, num_classes: int, hparams: Dict):
    super().__init__()
    self.save_hyperparameters(hparams)
    self.encoder = encoder
    self.prefix = prefix
    self.num_classes = num_classes
    self.classifier = nn.Linear(self.encoder.out_dim, self.num_classes)
    self.full_ft = self.prefix == "ft"
    self.metrics = AttrDict({
      "auroc": AUROC(num_classes=self.num_classes, average="macro", compute_on_step=False),
      "f1": AutoThresholdF1(num_classes=self.num_classes, average="macro", compute_on_step=False),
    })
  
  def forward(self, x: Tensor):
    return self.classifier(self.encoder(x))

  def _shared_step(self, batch: Tuple[Tensor, ...], stage: str):
    x, y = batch

    with torch.set_grad_enabled(self.full_ft): 
      z0 = self.encoder(x)
    y_hat = self.classifier(z0)
    loss = bce(y_hat, y)

    self.log(f"{self.prefix}_{stage}_loss", loss)
    if stage == "train": return loss
    
    y_probs = y_hat.softmax(dim=-1).float()
    y_long = y.long()
    for metric in self.metrics.values():
      metric(y_probs, y_long)
  
  def _shared_epoch_end(self, stage: str):
    for mn, m in self.metrics.items():
      try: 
        self.log(f"{self.prefix}_{stage}_{mn}", m.compute())
      except Exception as e: 
        print(f"Could not calculate {mn}: {e}")
      m.reset()

  def training_step(self, batch, batch_idx):
    return self._shared_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._shared_step(batch, "test")
  
  def on_test_start(self):
    self.metrics.f1.freeze()

  def validation_epoch_end(self, outputs):
    self._shared_epoch_end("val")
  
  def test_epoch_end(self, outputs):
    self._shared_epoch_end("test")
  
  def configure_optimizers(self):
    return Adam(self.parameters() if self.full_ft else self.classifier.parameters(), self.hparams.lr)