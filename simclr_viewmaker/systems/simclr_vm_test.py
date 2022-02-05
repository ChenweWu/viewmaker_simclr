import matplotlib
matplotlib.use('Agg')
from . import models
from .losses import SupConLoss
from .models import MLP
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from .models.encoders import viewmaker_1d
from torch.autograd import Function
import wandb
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from attrdict import AttrDict
from .metrics import AutoThresholdF1
from torchmetrics import AUROC
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from typing import Dict, Tuple
from .utils import configs as cfg, MODE
#from . import SimCLR_VM2
class SimCLR_VM(pl.LightningModule):
  def __init__(self, encoder: str, num_channels: int, hparams: Dict) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)
    
    self.num_channels = num_channels
    self.encoder = getattr(models, encoder)(self.num_channels)
    self.mlp = MLP(self.encoder.out_dim, out_dim=self.hparams.embedding_dim)
    self.loss_fn = SupConLoss(self.hparams.temp)
    self.viewmaker=self.create_viewmaker()
#     self.f_orig = "/home/chw5444/aihc-aut20-selfecg-main/3KG_PATH/data/val/orig.pt"
#     self.f_lbls = "/home/chw5444/aihc-aut20-selfecg-main/3KG_PATH/data/val/lbls.pt"
#     self.orig = torch.load(self.f_orig)[:64]
#     print("orig",self.orig.shape)
#     self.lbls = torch.load(self.f_lbls)[1][:64]
#     self.classifier = nn.Linear(self.encoder.out_dim, 23)
    
#     print("lbls",self.lbls.shape)
#     self.metrics = AttrDict({
#       "auroc": AUROC(num_classes=23, average="macro", compute_on_step=False),
#       "f1": AutoThresholdF1(num_classes=23, average="macro", compute_on_step=False),
#     })
  # self.revgrad=RevGrad()

  def forward(self, x: Tensor) -> Tensor:
    N, C, L = x.shape
    x = self._forward_impl(x.view(-1, self.num_channels, L)).view(N, C // self.num_channels, -1)
    return F.normalize(torch.mean(x, dim=1), dim=-1)

  def create_viewmaker(self):
    vmk=viewmaker_1d( num_channels=1,
            distortion_budget=0.01,
            clamp=False)
    return vmk
  

  
  def grad_reverse(self, x : Tensor) -> Tensor:
    return GradReverse.apply(x)
        
  def _forward_impl(self, x: Tensor) -> Tensor:
   # print("xshape",x.shape)
    inputs_to_log=x.detach()[0].view(-1,2500,1).cpu().numpy()
    x = self.viewmaker(x)
    views_to_log = x.detach()[0].view(-1,2500,1).cpu().numpy()
    x = self.grad_reverse(x)
#     x = self.revgrad(x)
    x = self.encoder(x)
    x = self.mlp(x)
    
    
    if self.global_step % (106467//64) == 0:
        diffs_to_log = views_to_log - inputs_to_log
        inputs = []
        augs = []
        diffs = []
        aug_3kg = []
        for view in diffs_to_log:
            f, ax = plt.subplots()
            ax.plot(np.arange(2500), view)
            ax.set_xlabel("t")
            ax.set_ylabel("Signal")
            ax.set_title("Viewmaker Perturbation Added to ECG 1D signal")
            diffs.append(wandb.Image(ax, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"))
            plt.close('all')
#             wandb.log({"diffs":diffs, "epoch": self.current_epoch})
        for view in views_to_log:
            f, ax = plt.subplots()
            ax.plot(np.arange(2500), view)
            ax.set_xlabel("t")
            ax.set_ylabel("Signal")
            ax.set_title("Viewmaker Augmented ECG 1D signal")
            augs.append(wandb.Image(ax, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"))
            plt.close('all') 
#             wandb.log({"augmented inputs":augs, "epoch": self.current_epoch})
        for view in inputs_to_log:
            f, ax = plt.subplots()
            ax.plot(np.arange(2500), view)
            ax.set_xlabel("t")
            ax.set_ylabel("Signal")
            ax.set_title("Input ECG 1D signal")
            inputs.append(wandb.Image(ax, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"))
            plt.close('all') 
#             wandb.log({"inputs" : inputs, "epoch": self.current_epoch})\
        wandb.log({"diffs":diffs,"augmented inputs":augs,"inputs" : inputs, "epoch": self.current_epoch})
        
#     if self.global_step % 5000 ==0:
#          ####
#         self.trainer.save_checkpoint(f"{self.global_step}.ckpt")
#         model = SimCLR_VM.load_from_checkpoint(f"{self.global_step}.ckpt", encoder="basic_cnn", num_channels=12, strict=False).enable_downstream()
#         x1, y1 = model.orig, model.lbls
#         with torch.no_grad():
#             z0 = model.encoder(x1)
#             y_hat = model.classifier(z0)
#             loss = bce(y_hat, y1)

#             self.log(f"{self.prefix}_{stage}_loss", loss)

#             y_probs = y_hat.softmax(dim=-1).float()
#             y_long = y.long()
#             for metric in model.metrics.values():
#               metric(y_probs, y_long)
    return x

        
  def update_view_boundary(self ):
    self.viewmaker.distortion_budget = 0.01
    
  def _shared_step(self, batch, stage: str) -> Tensor:
    *augs, pid = batch
    
    loss = self.loss_fn(tuple(self(aug) for aug in augs), pid)
    self.log(f"vm_{stage}_loss", loss)
   
    return loss
  
  def training_step(self, batch, batch_idx):
    *augs, pid = batch
    l=[tuple(self.forward(aug) for aug in augs), pid]
    return l

  def training_step_end(self, l):
    augs,pid=l
    loss = self.loss_fn(augs, pid)
    self.log(f"vm_train_loss", loss)
    wandb.log({"training loss":loss})
    self.update_view_boundary()
    return loss
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
  
class GradReverse(Function):
  @staticmethod  
  def forward(ctx,x):
        
    return x.view_as(x)
  @staticmethod 
  def backward(ctx, grad_output):
    output = grad_output.neg()

    return output
# from torch.autograd import Function


# class RevGrad(Function):
#     @staticmethod
#     def forward(ctx, input_, alpha_):
#         ctx.save_for_backward(input_, alpha_)
#         output = input_
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):  
#         grad_input = None
#         _, alpha_ = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = -grad_output * alpha_
#         return grad_input, None


# revgrad = RevGrad.apply

# class RevGrad(torch.nn.Module):
#     def __init__(self, alpha=1., *args, **kwargs):
#         """
#         A gradient reversal layer.
#         This layer has no parameters, and simply reverses the gradient
#         in the backward pass.
#         """
#         super().__init__(*args, **kwargs)

#         self._alpha = torch.tensor(alpha, requires_grad=False)

#     def forward(self, input_):
#         return revgrad(input_, self._alpha)
