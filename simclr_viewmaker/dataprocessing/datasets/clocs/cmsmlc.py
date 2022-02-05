from .clocs_interface import CLOCSDataset
from .cmsc import CMSCDataset
from ...transforms import MultiRandomTransform
import pandas as pd
from typing import Optional

class CMSMLCDataset(CLOCSDataset):
  def __init__(self, df: pd.DataFrame, prefix_path: str, 
               transform: Optional[MultiRandomTransform] = None):
    super().__init__(df, prefix_path, transform)

    view1, view2 = CMSCDataset.split_pid(self.lbls)
    self.orig = self.orig[view1].contiguous(), self.orig[view2].contiguous()
    self.lbls = self.lbls[view1].contiguous()
  
  def __getitem__(self, i):
    return *(lead for view in self.orig 
                  for aug in self.transform(view[i]) 
                  for lead in aug.split(1, dim=-2)), self.lbls[i]