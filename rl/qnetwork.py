import torch.nn as nn
import torch

class SolverLayer(nn.Module):
  def __init__(self, in_filters:int, hidden_filters:int, out_filters:int):
    super(SolverLayer, self).__init__()

    if in_filters != out_filters:
      self.skip = nn.Conv2d(in_filters, out_filters, kernel_size=1)
    else:
      self.skip = nn.Identity()

    self.initial_normalization = nn.InstanceNorm2d(in_filters) # Normalize every individual game within its filters. Not positive this is a good idea... maybe use the 3d version?

    self.HorizontalDependencies = self._get_dependency_module(in_filters, hidden_filters, (9,1))
    self.VerticalDependencies = self._get_dependency_module(in_filters, hidden_filters, (1,9))
    self.QuadrantDependencies = self._get_dependency_module(in_filters, hidden_filters, (3, 3), stride=3)

    num_hidden = hidden_filters * 3
    
    self.Reduce = nn.Sequential(
        nn.Conv2d(num_hidden, num_hidden, kernel_size=(3, 3), padding=1),
        nn.Conv2d(num_hidden, out_filters, kernel_size=(1, 1)), # Look at each cell only, without neighbors; neighbors have already been considered.
        nn.LeakyReLU()
    )

    self.Final = nn.LeakyReLU()

  def forward(self, x):
    skip = self.skip(x)
    x = self.initial_normalization(x)
    horizontal_result = self.HorizontalDependencies(x)
    vertical_result = self.VerticalDependencies(x)
    quadrant_result = self.QuadrantDependencies(x)

    combined = torch.cat((horizontal_result, vertical_result, quadrant_result), dim=1)

    reduced = self.Reduce(combined)

    residualized = reduced + skip
    return self.Final(residualized)

  def _get_dependency_module(self, in_filters, hidden_filters, kernel_size, stride=None):
    if stride is None:
      stride = 1
      
    return nn.Sequential(
        nn.Conv2d(in_filters, hidden_filters*9, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(hidden_filters*9),
        nn.ReLU(),
        nn.ConvTranspose2d(hidden_filters*9, hidden_filters, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(hidden_filters),
        nn.Dropout(),
        nn.ReLU()
    )

class QNetwork(nn.Module):
  def __init__(self, solver_depth:int, action_size:int, growth_rate=(3,9)):
    super(QNetwork, self).__init__()
    self.output_size = action_size
    if type(growth_rate) == int:
      growth_base = 3
      growth_frequency = growth_rate
    else:
      growth_base, growth_frequency = growth_rate
    layer_numbers = [(growth_base**(i // growth_frequency + 2), growth_base**( (i+1) // growth_frequency + 2)) for i in range(solver_depth - 1)]
    reducer_layer = layer_numbers[-1][1]
    self.net = nn.Sequential(
        *[ SolverLayer(i, i, o) for i, o in layer_numbers],
        SolverLayer(reducer_layer, reducer_layer, 9),
    )

  def forward(self, x):
    return self.net(x)

    