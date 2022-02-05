import torch
from torch import Tensor

_INVERSE_DOWERS = torch.tensor(
  [[-0.172, -0.074,  0.122,  0.231, 0.239, 0.194,  0.156, -0.010],  #x
   [ 0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227,  0.887],  #y
   [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108,  0.022,  0.102]]  #z
)  

_DOWERS = torch.tensor(
  [[ 0.632, -0.235,  0.059],  #I
   [ 0.235,  1.066, -0.132],  #II
   [-0.397,  1.301, -0.191],  #III
   [-0.434, -0.415,  0.037],  #aVR
   [ 0.515, -0.768,  0.125],  #aVL
   [-0.081,  1.184, -0.162],  #aVF
   [-0.515,  0.157, -0.917],  #V1
   [ 0.044,  0.164, -1.387],  #V2
   [ 0.882,  0.098, -1.277],  #V3
   [ 1.213,  0.127, -0.601],  #V4
   [ 1.125,  0.127, -0.086],  #V5
   [ 0.831,  0.076,  0.230]]  #V6
) 

_FRANK_LEAD_INDICES = [6, 7, 8, 9, 10, 11, 0, 1]
assert len(_FRANK_LEAD_INDICES) == _INVERSE_DOWERS.shape[1]

def to_vcg(ecg: Tensor) -> Tensor:
  return torch.matmul(_INVERSE_DOWERS, ecg[..., _FRANK_LEAD_INDICES, :])

def to_ecg(vcg: Tensor) -> Tensor:
  return torch.matmul(_DOWERS, vcg.contiguous())
