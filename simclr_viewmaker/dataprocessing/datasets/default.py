from ..transforms import MultiRandomTransform

import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate, resample
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Tuple
from utils import all_paths_exist, CLASSES, NUM_CLASSES, REC_FREQ, RAW_DATA_PATH, REC_DIMS, MODE

class PhysionetDataset(Dataset):
  def __init__(self, df: pd.DataFrame, prefix_path: str, 
               transform: Optional[MultiRandomTransform]):
    os.makedirs(prefix_path, exist_ok=True)
    f_orig = f"{prefix_path}/orig.pt"
    f_lbls = f"{prefix_path}/lbls.pt"
    f_rdim = f"{prefix_path}/rdim.pt"

    pretrain = (MODE in ["pt","vm"])
    self.transform = transform if pretrain and transform else lambda x: (x,)

    if all_paths_exist(f_orig, f_lbls, f_rdim) and torch.load(f_rdim) == REC_DIMS:
      self.orig = torch.load(f_orig)
      self.lbls = torch.load(f_lbls)[0 if pretrain else 1]
    else:
      rC, rL = REC_DIMS
      start_N = 2 ** 14
      self.orig = torch.empty(start_N, rC, rL)
      pid = torch.empty(start_N, dtype=torch.long)
      dx = torch.empty(start_N, NUM_CLASSES)

      curr_r = 0
      for i in tqdm(range(len(df))):
        recording = PhysionetDataset._read_recording(df.iloc[i]["Patient"], REC_DIMS)
        if recording is None: continue

        N = len(recording)
        while curr_r + N >= len(self.orig): 
          self.orig = torch.cat((self.orig, torch.empty_like(self.orig)), dim=0)
          pid = torch.cat((pid, torch.empty_like(pid)), dim=0)
          dx = torch.cat((dx, torch.empty_like(dx)), dim=0)
        
        self.orig[curr_r:curr_r + N] = recording
        pid[curr_r:curr_r + N] = i
        dx[curr_r:curr_r + N] = torch.from_numpy(df.iloc[i][CLASSES].astype(np.int).to_numpy()).unsqueeze_(0)
        curr_r += N
      
      self.orig = self.orig[:curr_r].contiguous()
      pid = pid[:curr_r].contiguous()
      dx = dx[:curr_r].contiguous()

      torch.save(self.orig, f_orig)
      torch.save((pid, dx), f_lbls)
      torch.save(REC_DIMS, f_rdim)

      self.lbls = pid if pretrain else dx
  
  def __getitem__(self, i):
    return *self.transform(self.orig[i]), self.lbls[i]
  
  def __len__(self):
    return len(self.lbls)

  def apply_mask(self, mask: np.ndarray):
    self.orig = self.orig[mask].contiguous()   
    self.lbls = self.lbls[mask].contiguous()

  @staticmethod
  def _read_recording(id: str, rdim: Tuple) -> Optional[Tensor]:
    file_name = f"{RAW_DATA_PATH}/{id}"
    rC, rL = rdim

    recording = PhysionetDataset._process_recording(file_name, rL)

    C, L = recording.shape
    if C != rC or L < rL or not torch.all(recording.isfinite()): return None

    recording = recording[:, :rL * (L // rL)].view(C, -1, rL).transpose(0, 1) #exhaustive crop
    return recording.contiguous()

  @staticmethod
  def _process_recording(file_name: str, rL: int):
    recording = loadmat(f"{file_name}.mat")['val'].astype(float)

    # Standardize sampling rate
    sampling_rate = PhysionetDataset._get_sampling_rate(file_name)

    if sampling_rate > REC_FREQ:
      recording = np.copy(decimate(recording, int(sampling_rate / REC_FREQ)))
    elif sampling_rate < REC_FREQ:
      recording = np.copy(resample(recording, int(recording.shape[-1] * (REC_FREQ / sampling_rate)), axis=1))
    
    return torch.from_numpy(PhysionetDataset._normalize(recording))
  
  @staticmethod
  def _normalize(x: np.ndarray):
    return x / (np.max(x) - np.min(x))
  
  @staticmethod
  def _get_sampling_rate(file_name: str):
    with open(f"{file_name}.hea", 'r') as f:
      return int(f.readline().split(None, 3)[2])
