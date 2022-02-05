from attrdict import AttrDict
import glob
import os
import pprint as pp
import torch
from torch import Tensor
from systems import SimCLR, DownstreamTask
from systems.metrics import AutoThresholdF1
from torchmetrics import AUROC, BootStrapper, Metric
from utils import get_num_encoder_channels, NUM_CLASSES, CLASS_LABELS

QUANTILES = torch.tensor([0.025, 0.975])

def load_all_checkpoints(system: str, encoder: str, dir: str): 
  ckpts = glob.glob(f"{dir}/*.ckpt")
  ckpts.sort(key=os.path.getmtime)

  return [
    DownstreamTask.load_from_checkpoint(
      checkpoint_path=ckpt, 
      strict=False,
      encoder=SimCLR(
        encoder=encoder,
        num_channels=get_num_encoder_channels(system),
        hparams={
          "lr": 0,
          "temp": 0,
          "embedding_dim": 256
        }
      ).enable_downstream(),
      prefix="bs",
      num_classes=NUM_CLASSES
    ).eval() for ckpt in ckpts
  ]

def initialize_metric(name: str, per_class: bool):
  metric = None
  average = "macro" if not per_class else None
  if name == "auroc":
    metric = AUROC(
      num_classes=NUM_CLASSES,
      compute_on_step=False, 
      average=average,
    )
  elif name == "f1":
    metric = AutoThresholdF1(
      num_classes=NUM_CLASSES,
      compute_on_step=False,
      average=average,
    )
  else:
    raise NotImplementedError(f"Metric {name} isn't supported.")
  
  return metric
  
def bootstrap_metric(metric: Metric):
  return BootStrapper(
    metric,
    num_bootstraps=1000,
    mean=False,
    std=False,
    raw=True,
    sampling_strategy="multinomial",
    compute_on_step=False,
  )

def get_result_message(result: AttrDict) -> str:
  def parse_tensor(t: Tensor):
    if t.shape and t.shape != QUANTILES.shape: #per class
      if t.shape[-1] != NUM_CLASSES or len(t.shape) not in [1, 2]:
        raise ValueError(f"Cannot parse tensor result with shape {t.shape}.")
      
      return dict(zip(CLASS_LABELS, t.T))
      
    else: #average
      return t

  if isinstance(result, Tensor): #non-bootstrapped
    return pp.pformat(parse_tensor(result), indent=2)
  elif isinstance(result, AttrDict): #bootstrapped:
    for key, val in result.items():
      if not isinstance(val, Tensor):
        raise ValueError(f"Cannot parse result key {key} with type {val.__class__.__name__}")
      
      if key in ["mean", "std", "quantile"]:
        result[key] = pp.pformat(parse_tensor(val), indent=2)
    return f"MEAN\n{result.mean}\n\nSTD\n{result.std}\n\nQUANTILES {QUANTILES}\n{result.quantile}"
  else:
    raise ValueError(f"Cannot parse result with type {result.__class__.__name__}")

def print_all_messages(messages):
  print("\n\n--------------------------------------------\n\n".join(messages))

def calculate_bootstrapped_stats(raw: Tensor): #necessary fn bc of a bug in torchmetrics
  return AttrDict({
    "mean": raw.mean(dim=0),
    "std": raw.std(dim=0),
    "quantile": raw.quantile(QUANTILES, dim=0)
  })