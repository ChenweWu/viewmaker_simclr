from ..default import PhysionetDataset
from ...transforms import MultiRandomTransform
import pandas as pd
from typing import Optional
from utils import MODE

class CLOCSDataset(PhysionetDataset):  
  def __init__(self, df: pd.DataFrame, prefix_path: str, 
               transform: Optional[MultiRandomTransform] = None):
    if MODE != "pt": raise ValueError(f"Cannot use {self.__class__.__name__} on downstream task.")
    super().__init__(df, prefix_path, transform)