from attrdict import AttrDict
from dataprocessing import PhysionetDataModule
from .helpers import *
import torch
from torch import Tensor
from torchmetrics import Metric
from tqdm import tqdm
from utils import CLASS_LABELS

class MetricDifference(Metric):
  def __init__(self, base_metric: Metric, compute_on_step=False, dist_sync_on_step=False):
    super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
    
    self.base = base_metric.clone()
    self.comp = base_metric.clone()
  
  def update(self, b: Tensor, c: Tensor, y: Tensor):
    if b.shape != c.shape: raise ValueError(
      "Tensors passed into Difference metric must be identically shaped, "
     f"instead got shapes {b.shape} for 1st arg and {c.shape} for 2nd arg"
    )
    
    self.base(b, y)
    self.comp(c, y)
  
  def compute(self):
    return self.comp.compute() - self.base.compute()
  
  def reset(self) -> None:
    self.base.reset()
    self.comp.reset()
  
  @property
  def is_differentiable(self):
    return self.base.is_differentiable and self.comp.is_differentiable

def bootstrap(cfg: AttrDict):
  dm = PhysionetDataModule("PhysionetDataset", None, cfg.batch_size, cfg.num_workers)
  dm.setup("test")

  baseline_models = load_all_checkpoints(cfg.baseline.system, cfg.baseline.encoder, cfg.baseline.dir)
  output_messages = []
  for comparison in tqdm(cfg.comparisons):
    bootstrapper = bootstrap_metric(MetricDifference(initialize_metric(cfg.metric, cfg.per_class)))
    comparison_models = load_all_checkpoints(comparison.system, comparison.encoder, comparison.dir)

    with torch.no_grad():
      for x, y in tqdm(dm.test_dataloader(), leave=False):
        for model_idx in tqdm(range(min(len(baseline_models), len(comparison_models))), leave=False):
          bootstrapper(
            baseline_models[model_idx](x).softmax(dim=-1), 
            comparison_models[model_idx](x).softmax(dim=-1), 
            y.long()
          )

    result = calculate_bootstrapped_stats(bootstrapper.compute()["raw"])
    l, r = result.quantile

    significance_message = None
    if cfg.per_class:
      significance_message = "\n".join([
        f"{label}: {comparison.experiment} is significantly {assess_interval(l[i], r[i])}" 
        for i, label in enumerate(CLASS_LABELS)
      ])
    else:
      significance_message = (
        f"{comparison.experiment} is significantly {assess_interval(l,r)}"
      )

    output_messages.append(
      f"COMPARISON RESULTS: {cfg.baseline.experiment} VS. {comparison.experiment}\n"
      f"{significance_message}\n\n{get_result_message(result)}"
    )
  
  print_all_messages(output_messages)

def assess_interval(l, r):
  if l > 0: return "BETTER"
  elif r < 0: return "WORSE"
  else: return "NO DIFFERENT"
