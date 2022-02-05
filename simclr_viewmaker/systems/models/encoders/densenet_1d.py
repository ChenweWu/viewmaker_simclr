import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple, Union


__all__ = ["densenet121", "densenet169", "densenet201", "densenet161"]


class _DenseLayer(nn.Module):
  def __init__(self, num_input_features: int, growth_rate: int,
               bn_size: int, drop_rate: float, memory_efficient: bool = False) -> None:
    super().__init__()
    self.norm1 = nn.BatchNorm1d(num_input_features)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

    self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    self.dropout = nn.Dropout(float(drop_rate), inplace=True)

  # todo: rewrite when torchscript supports any
  def any_requires_grad(self, input: List[Tensor]) -> bool:
    for tensor in input:
      if tensor.requires_grad:
        return True
    return False

  # torchscript does not yet support *args, so we overload method
  # allowing it to take either a List[Tensor] or single Tensor
  def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:  # noqa: F811
    if isinstance(x, list):
      x = torch.cat(x, dim=1)
    
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.conv1(x)

    x = self.norm2(x)
    x = self.relu2(x)
    x = self.conv2(x)

    x = self.dropout(x)

    return x


class _DenseBlock(nn.ModuleDict):
  def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
               growth_rate: int, drop_rate: float) -> None:
    super().__init__()
    for i in range(num_layers):
      self.add_module(f"denselayer{i+1}", 
        _DenseLayer(
          num_input_features + i * growth_rate,
          growth_rate=growth_rate,
          bn_size=bn_size,
          drop_rate=drop_rate
        )
      )

  def forward(self, init_features: Tensor) -> Tensor:
    features = [init_features]
    for layer in self.values():
      features.append(layer(features))
    return torch.cat(features, dim=1)


class _Transition(nn.Module):
  def __init__(self, num_input_features: int, num_output_features: int) -> None:
    super().__init__()
    self.norm = nn.BatchNorm1d(num_input_features)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
    self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

  def forward(self, x: Tensor):
    x = self.norm(x)
    x = self.relu(x)
    x = self.conv(x)
    x = self.pool(x)


class DenseNet(nn.Module):
  r"""Densenet-BC model class, based on
  `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

  Args:
    growth_rate (int) - how many filters to add each layer (`k` in paper)
    block_config (list of 4 ints) - how many layers in each pooling block
    num_init_features (int) - the number of filters to learn in the first convolution layer
    bn_size (int) - multiplicative factor for number of bottle neck layers
      (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float) - dropout rate after each dense layer
    num_classes (int) - number of classification classes
    memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
      but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
  """

  def __init__(self, input_channels: int, output_channels: int = 256, growth_rate: int = 32, 
               block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
               num_init_features: int = 64, bn_size: int = 4, drop_rate: float = 0) -> None:

    super(DenseNet, self).__init__()

    self.in_dim = input_channels
    self.out_dim = output_channels
    # First convolution
    self.features = nn.Sequential(
      nn.Conv1d(input_channels, num_init_features, kernel_size=7, stride=2,
                padding=3, bias=False),
      nn.BatchNorm1d(num_init_features),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    )

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(
        num_layers=num_layers,
        num_input_features=num_features,
        bn_size=bn_size,
        growth_rate=growth_rate,
        drop_rate=drop_rate
      )
      self.features.add_module(f"denseblock{i+1}", block)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module(f"transition{i+1}", trans)
        num_features = num_features // 2

    # Linear layer
    self.classifier = nn.Linear(num_features, output_channels)

    # Official init from torch repo.
    for m in self.modules():
      if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)

  def forward(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool1d(out, 1)
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out


def _densenet(input_channels: int, growth_rate: int, block_config: Tuple[int, int, int, int], 
              num_init_features: int) -> DenseNet:
  return DenseNet(
    input_channels, 
    growth_rate=growth_rate, 
    block_config=block_config, 
    num_init_features=num_init_features
  )


def densenet121(input_channels: int) -> DenseNet:
  return _densenet(input_channels, 32, (6, 12, 24, 16), 64)


def densenet161(input_channels: int) -> DenseNet:
  return _densenet(input_channels, 48, (6, 12, 36, 24), 96)


def densenet169(input_channels: int) -> DenseNet:
  return _densenet(input_channels, 32, (6, 12, 32, 32), 64)


def densenet201(input_channels: int) -> DenseNet:
  return _densenet(input_channels, 32, (6, 12, 48, 32), 64)
