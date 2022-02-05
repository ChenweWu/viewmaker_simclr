from . import datasets
from .transforms import MultiRandomTransform
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from typing import Optional
from utils import EQUIV_CLASS_GROUPS, FOLDS, REC_DIMS, \
                  SAVED_DATA_PATH, SPLITS_PATH

class PhysionetDataModule(pl.LightningDataModule):
  def __init__(self, dataset_type: str, transform: Optional[MultiRandomTransform], 
               batch_size: int, num_workers: int):
    super().__init__()

    self.ds_type = getattr(datasets, dataset_type)
    self.transform = transform
    self.splits = PhysionetDataModule._get_splits()
    self.dims = (1, *REC_DIMS)
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def setup(self, stage: Optional[str] = None) -> None:
    if stage in ["fit", None]:
      self._load_dataset("train")
    if stage in ["fit", "predict", "validate", None]:
      self._load_dataset("val")
    if stage in ["predict", "test", None]:
      self._load_dataset("test")

  def _load_dataset(self, stage: str):
    df = self.splits[self.splits["fold"].isin(FOLDS[stage])]
    print(f"Loading {stage} dataset...", end='')
    setattr(self, stage, self.ds_type(df, f"{SAVED_DATA_PATH}/{stage}", self.transform))
    print(f"done, dataset has {len(getattr(self, stage))} samples.")
  
  def _shared_dataloader(self, stage: str):
    return DataLoader(getattr(self, stage), batch_size=self.batch_size, num_workers=self.num_workers, 
                      pin_memory=True, shuffle=stage == "train")

  def fractionate(self, frac: float, split: int) -> None:
    if not hasattr(self, "train"): self.setup("fit")
    print("Fractionating training dataset...", end='')
    N = len(self.train)
    seed_everything(split)
    mask = np.random.choice(N, size=int(frac * N), replace=False)
    seed_everything(6)
    self.train.apply_mask(mask)
    print(f"done, training dataset now has {len(self.train)} samples")
  
  def train_dataloader(self) -> DataLoader:
    return self._shared_dataloader("train")
  
  def val_dataloader(self) -> DataLoader:
    return self._shared_dataloader("val")

  def test_dataloader(self) -> DataLoader:
    return self._shared_dataloader("test")
  
  @staticmethod
  def _get_splits():
    splits = pd.read_csv(SPLITS_PATH, index_col=0)
    for dx, dup_dxs in EQUIV_CLASS_GROUPS.items():
      for dup in dup_dxs:
        splits[dx] |= splits[dup]
      splits = splits.drop(columns=dup_dxs)
    return splits