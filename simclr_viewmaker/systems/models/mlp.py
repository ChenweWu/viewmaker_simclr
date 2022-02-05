from torch import nn, Tensor

class MLP(nn.Module):
  def __init__(self, in_dim: int, out_dim: int = 128) -> None:
    super().__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim

    self.relu = nn.LeakyReLU(0.2, inplace=True)

    self.fc1 = nn.Linear(self.in_dim, self.out_dim, bias=False)
    self.bn1 = nn.BatchNorm1d(self.out_dim)

    self.fc2 = nn.Linear(self.out_dim, self.out_dim)
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.relu(x)

    x = self.fc1(x)
    x = self.relu(x)
    x = self.bn1(x)

    x = self.fc2(x)
    return x