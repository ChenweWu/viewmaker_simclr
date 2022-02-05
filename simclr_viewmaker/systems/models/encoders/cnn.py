from torch import nn

__all__ = ["basic_cnn"]

class BasicCNN(nn.Sequential):
  def __init__(self, C_in, C_out: int = 256) -> None:
    self.in_dim = C_in
    self.out_dim = C_out
    super().__init__(
      *BasicCNN.make_block(C_in, 16, kernel_size=7, stride=4),
      *BasicCNN.make_block(16, 32, kernel_size=7, stride=3),
      *BasicCNN.make_block(32, 64, kernel_size=5, stride=2),
      *BasicCNN.make_block(64, 64),
      *BasicCNN.make_block(64, 128),
      *BasicCNN.make_block(128, C_out),
      nn.AdaptiveAvgPool1d(1),
      nn.Flatten()
    )

  @staticmethod
  def make_block(C_in: int, C_out: int, kernel_size: int = 3,stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      nn.BatchNorm1d(C_out)
    )

def basic_cnn(C_in: int):
  return BasicCNN(C_in)