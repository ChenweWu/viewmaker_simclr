import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct
from .vcg import to_vcg, to_ecg

class _Augmentation(nn.Module):
  def __init__(self, no_op: bool) -> None:
    super().__init__()
    self._no_op = no_op

class _RandomVCGAugmentation(_Augmentation):
  def __init__(self, no_op: bool) -> None: super().__init__(no_op)

class _RandomECGAugmentation(_Augmentation):
  def __init__(self, no_op: bool) -> None: super().__init__(no_op)

class MultiRandomTransform(nn.Module):
  def __init__(self, nviews: int, args: Tuple[_Augmentation]) -> None:
    super().__init__()

    vcg = []
    ecg = []
    for arg in args:
      if   isinstance(arg, _RandomVCGAugmentation) and not arg._no_op: vcg.append(arg)
      elif isinstance(arg, _RandomECGAugmentation) and not arg._no_op: ecg.append(arg)
    self.vcg = nn.Sequential(*vcg)
    self.ecg = nn.Sequential(*ecg)
    self.nviews = nviews
  
  def forward(self, x: Tensor):
    with torch.no_grad():
      if len(x.shape) == 2: 
        x = x.unsqueeze(0)
      elif len(x.shape) != 3: 
        raise ValueError(f"Input tensor to {self.__class__.__name__} must be 2 or 3 dimensions.")
      return tuple(self._forward_impl(x).squeeze_(0) for _ in range(self.nviews))

  def _forward_impl(self, x: Tensor) -> Tensor:
    if self.vcg: x = to_ecg(self.vcg(to_vcg(x)))
    elif self.ecg: x = x.clone()
    
    return self.ecg(x)

class SingleRandomTransform(MultiRandomTransform):
  def __init__(self, *args: _Augmentation) -> None:
    super().__init__(1, args)

class DoubleRandomTransform(MultiRandomTransform):
  def __init__(self, *args: _Augmentation) -> None:
    super().__init__(2, args)

class DoubleVMTransform(MultiViewMakerTransform):
  def __init__(self, *args: _Augmentation) -> None:
    super().__init__(2, args)


class RandomGaussian(_RandomECGAugmentation):
  def __init__(self, do_something: bool) -> None:
    super().__init__(not do_something)
  
  def forward(self, x: Tensor) -> Tensor:
    return x + torch.randn_like(x)

class RandomRotation(_RandomVCGAugmentation):
  R = np.array([
    lambda theta: torch.tensor([[1,             0,              0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta),  np.cos(theta)]]),
    lambda theta: torch.tensor([[np.cos(theta),  0, np.sin(theta)],
                                [0,              1,             0],
                                [-np.sin(theta), 0, np.cos(theta)]]),
    lambda theta: torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
  ])

  def __init__(self, max_angle: float) -> None:
    super().__init__(np.isclose(max_angle, 0))
    self.max_angle = np.deg2rad(max_angle)
    self.angle_range = self.max_angle * 2

  def forward(self, x: Tensor) -> Tensor:
    return torch.matmul(self._get_rotations(len(x)), x)
  
  def _get_rotations(self, N: int) -> Tuple[Tensor, Tensor]:
    perm_fns = RandomRotation.R[np.vstack(tuple(np.random.permutation(3) for _ in range(N)))]
    thetas = torch.rand(N, 3).mul_(self.angle_range).sub_(self.max_angle)

    return torch.stack(tuple(
      map(lambda f, a: f[0](a[0]) @ f[1](a[1]) @ f[2](a[2]), perm_fns, thetas)))

class RandomScale(_RandomVCGAugmentation):
  def __init__(self, scale: float) -> None:
    super().__init__(np.isclose(scale, 1))
    self.scale = scale

  def forward(self, x: Tensor) -> Tensor:
    return torch.matmul(self._get_scales(len(x)), x)
  
  def _get_scales(self, N: int) -> Tensor:
    s = torch.from_numpy(np.random.uniform(1, self.scale, size=(N, 3))).float()
    mask = torch.rand_like(s) < 0.5
    s[mask] = 1 / s[mask]
    return torch.diag_embed(s)

class RandomChannelMask(_RandomECGAugmentation):
  def __init__(self, p: float) -> None:
    if p < 0 or p > 1: raise ValueError(f"p not within [0, 1]: saw {p}")
    super().__init__(np.isclose(p, 0))
    self.p = p
  
  def forward(self, x: Tensor) -> Tensor:
    x[self._get_mask(*x.shape[:2])]  = 0
    return x
  
  def _get_mask(self, N: int, C: int):
    mask = torch.zeros(N, C, dtype=torch.bool)
    for i, channels in enumerate(torch.multinomial(torch.ones(N, C), int(self.p * C))):
      mask[i, channels] = True
    return mask

class RandomTimeMask(_RandomECGAugmentation):
  def __init__(self, p: float) -> None:
    if p < 0 or p > 1: raise ValueError(f"p not within [0, 1]: saw {p}")
    super().__init__(np.isclose(p, 0))
    self.p = p

  def forward(self, x: Tensor) -> Tensor:
    N, C, L = x.shape

    mask_len = int(L * self.p)
    for n in range(N):
      for c in range(C):
        start = np.random.randint(L)
        stop = start + mask_len
        if stop >= L:
          mod = stop - L
          x[n, c, start:L]    = 0
          x[n, c,      :mod]  = 0
        else: 
          x[n, c, start:stop] = 0
    
    return x


'''Core architecture and functionality of the viewmaker network.

Adapted from the transformer_net.py example below, using methods proposed in Johnson et al. 2016

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''


ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}

class MultiViewMakerTransform(nn.Module):
  def __init__(self, nviews: int) -> None:
    super().__init__()

#     vcg = []
#     ecg = []
#     for arg in args:
#       if   isinstance(arg, _RandomVCGAugmentation) and not arg._no_op: vcg.append(arg)
#       elif isinstance(arg, _RandomECGAugmentation) and not arg._no_op: ecg.append(arg)
#     self.vcg = nn.Sequential(*vcg)
#     self.ecg = nn.Sequential(*ecg)
    self.nviews = nviews
#     self.view = self.create_viewmaker()
    
#   def create_viewmaker(self):
#         view_model = viewmaker.Viewmaker(
#             num_channels=1,
#             distortion_budget=0.05,
#             clamp=False,
#         )
#         return view_model
  
  def forward(self, x: Tensor):
    with torch.no_grad():
      if len(x.shape) == 2: 
        x = x.unsqueeze(0)
      elif len(x.shape) != 3: 
        raise ValueError(f"Input tensor to {self.__class__.__name__} must be 2 or 3 dimensions.")
      return tuple(self._forward_impl(x).squeeze_(0) for _ in range(self.nviews))

  def _forward_impl(self, x: Tensor) -> Tensor:
#     if self.vcg: x = to_ecg(self.vcg(to_vcg(x)))
#     elif self.ecg: x = x.clone()
    
#     return self.ecg(x)
#     return self.view(x)
    return x

# class Viewmaker(torch.nn.Module):
#     '''Viewmaker network that stochastically maps a multichannel 1D input to an output of the same size.'''
#     def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',  
#                 clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3):
#         '''Initialize the Viewmaker network.

#         Args:
#             num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
#                 Input will have shape [batch_size, num_channels, height, width]
#             distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
#                 Controls how strong the perturbations can be.
#             activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
#             clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
#             frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
#                 This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
#             downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
#                 higher-resolution inputs, but not evaluaed in the paper.
#             num_res_blocks: Number of residual blocks to use in the network.
#         '''
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.num_res_blocks = num_res_blocks
#         self.activation = activation
#         self.clamp = clamp
#         self.frequency_domain = frequency_domain
#         self.downsample_to = downsample_to 
#         self.distortion_budget = distortion_budget
#         self.act = ACTIVATIONS[activation]()

#         # Initial convolution layers (+ 1 for noise filter)
#         self.conv1 = ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1)
#         self.in1 = torch.nn.InstanceNorm1d(32, affine=True)
#         self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
#         self.in2 = torch.nn.InstanceNorm1d(64, affine=True)
#         self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
#         self.in3 = torch.nn.InstanceNorm1d(128, affine=True)
        
#         # Residual layers have +N for added random channels
#         self.res1 = ResidualBlock(128 + 1)
#         self.res2 = ResidualBlock(128 + 2)
#         self.res3 = ResidualBlock(128 + 3)
#         self.res4 = ResidualBlock(128 + 4)
#         self.res5 = ResidualBlock(128 + 5)
        
#         # Upsampling Layers
#         self.deconv1 = UpsampleConvLayer(128 + self.num_res_blocks, 64, kernel_size=3, stride=1, upsample=2)
#         self.in4 = torch.nn.InstanceNorm1d(64, affine=True)
#         self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
#         self.in5 = torch.nn.InstanceNorm1d(32, affine=True)
#         self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)

#     @staticmethod
#     def zero_init(m):
#         if isinstance(m, (nn.Linear, nn.Conv1d)):
#             # actual 0 has symmetry problems
#             init.normal_(m.weight.data, mean=0, std=1e-4)
#             # init.constant_(m.weight.data, 0)
#             init.constant_(m.bias.data, 0)
#         elif isinstance(m, nn.BatchNorm1d):
#             pass

#     def add_noise_channel(self, x, num=1, bound_multiplier=1):
#         # bound_multiplier is a scalar or a 1D tensor of length batch_size
#         batch_size = x.size(0)
# #         print("batch size", batch_size)
#         filter_size = x.size(-1)
# #         print("filter size", filter_size)
#         shp = (batch_size, num, filter_size)
#         bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
#         noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
# #         print("x shape", x.shape)
# #         print("noise shape", noise.shape)
#         return torch.cat((x, noise), dim=1)

#     def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
#         if num_res_blocks not in list(range(6)):
#             raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

#         y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
#         y = self.act(self.in1(self.conv1(y)))
#         y = self.act(self.in2(self.conv2(y)))
#         y = self.act(self.in3(self.conv3(y)))

#         # Features that could be useful for other auxilary layers / losses.
#         # [batch_size, 128]
#         features = y.clone().mean([-1, -2])
        
#         for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
#             if i < num_res_blocks:
#                 y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))

#         y = self.act(self.in4(self.deconv1(y)))
#         y = self.act(self.in5(self.deconv2(y)))
#         y = self.deconv3(y)

#         return y, features
    
#     def get_delta(self, y_pixels, eps=1e-4):
#         '''Constrains the input perturbation by projecting it onto an L1 sphere'''
#         distortion_budget = self.distortion_budget
#         delta = torch.tanh(y_pixels) # Project to [-1, 1]
# #         print("delta", delta)
# #         print(delta.abs().mean([1,2], keepdim=True))
#         avg_magnitude = delta.abs().mean([1,2,], keepdim=True)
#         max_magnitude = distortion_budget
#         delta = delta * max_magnitude / (avg_magnitude + eps)
#         return delta

#     def forward(self, x):
#         if self.downsample_to:
#             # Downsample.
#             x_orig = x
#             x = torch.nn.functional.interpolate(
#                 x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
#         y = x
        
#         if self.frequency_domain:
#             # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
#             # Uses the Discrete Cosine Transform.
#             # shape still [batch_size, C, W, H]
#             y = dct.dct_1d(y)

#         y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
#         delta = self.get_delta(y_pixels)
#         if self.frequency_domain:
#             # Compute inverse DCT from frequency domain to time domain.
#             delta = dct.idct_1d(delta)
#         if self.downsample_to:
#             # Upsample.
#             x = x_orig
#             delta = torch.nn.functional.interpolate(delta, size=x_orig.shape[-2:], mode='bilinear')

#         # Additive perturbation
#         result = x + delta
#         if self.clamp:
#             result = torch.clamp(result, 0, 1.0)

#         return result


# # ---

# class ConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
#         self.conv1d = torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv1d(out)
#         return out


# class ResidualBlock(torch.nn.Module):
#     """ResidualBlock
#     introduced in: https://arxiv.org/abs/1512.03385
#     recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
#     """

#     def __init__(self, channels, activation='relu'):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in1 = torch.nn.InstanceNorm1d(channels, affine=True)
#         self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in2 = torch.nn.InstanceNorm1d(channels, affine=True)
#         self.act = ACTIVATIONS[activation]()

#     def forward(self, x):
#         residual = x
#         out = self.act(self.in1(self.conv1(x)))
#         out = self.in2(self.conv2(out))
#         out = out + residual
#         return out


# class UpsampleConvLayer(torch.nn.Module):
#     """UpsampleConvLayer
#     Upsamples the input and then does a convolution. This method gives better results
#     compared to ConvTranspose2d.
#     ref: http://distill.pub/2016/deconv-checkerboard/
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpsampleConvLayer, self).__init__()
#         self.upsample = upsample
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
#         self.conv1d = torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         x_in = x
#         if self.upsample:
#             x_in = torch.nn.functional.interpolate(
#                 x_in, mode='nearest', scale_factor=self.upsample)
#         out = self.reflection_pad(x_in)
#         out = self.conv1d(out)
#         return out
    
    
# def viewmaker(num_channels: int, distortion_budget: int):
#   return Viewmaker(num_channels, distortion_budget)
