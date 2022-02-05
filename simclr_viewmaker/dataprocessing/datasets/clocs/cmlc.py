from .clocs_interface import CLOCSDataset
from ...transforms import MultiRandomTransform
import pandas as pd
from typing import Optional

class CMLCDataset(CLOCSDataset):  
  def __init__(self, df: pd.DataFrame, prefix_path: str, 
               transform: Optional[MultiRandomTransform] = None):
    super().__init__(df, prefix_path, transform)

  def __getitem__(self, i):
    *augs, lbls = super().__getitem__(i)
    return *(lead for aug in augs for lead in aug.split(1, dim=-2)), lbls