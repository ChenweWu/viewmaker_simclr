from .clocs_interface import CLOCSDataset
import pandas as pd
from ...transforms import MultiRandomTransform
from torch import Tensor
import torch
from typing import Optional
from utils import REC_DIMS

class CMSCDataset(CLOCSDataset):
  def __init__(self, df: pd.DataFrame, prefix_path: str, 
               transform: Optional[MultiRandomTransform] = None):
    if transform.vcg: raise NotImplementedError("CMSC does not support VCG augmentations.")
    super().__init__(df, prefix_path, transform)

    rC, rL = REC_DIMS
    view1, view2 = CMSCDataset.split_pid(self.lbls)
    self.orig = (self.orig[view1].contiguous().view(-1, 1, rL), 
                 self.orig[view2].contiguous().view(-1, 1, rL))

    self.lbls = self.lbls[view1].contiguous().repeat_interleave(rC) #view1 and view2 have same pids
    for i in range(len(self.lbls)):
      self.lbls[i] = hash((self.lbls[i], i % rC))
  
  def __getitem__(self, i):
    return *(aug for view in self.orig for aug in self.transform(view[i])), self.lbls[i]

  @staticmethod
  def split_pid(pids: Tensor):
    view1 = []
    view2 = []
    for pid in torch.unique(pids):
      crop_mask = pid == pids
      pid_indices = torch.nonzero(crop_mask).flatten()
      if len(pid_indices) % 2 == 1: #i.e. odd number of crops
        pid_indices = pid_indices[:-1]
      
      N = len(pid_indices)
      if N == 0: continue
      view1.extend(pid_indices[:N // 2])
      view2.extend(pid_indices[N // 2:])
    
    return torch.tensor(view1), torch.tensor(view2)