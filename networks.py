from utils import partial_class
from torch import nn


class SwishN(nn.Module):
    def __init__(self, order=1):
        super(SwishN, self).__init__()
        self.order = order

    def __call__(self, x):
        return torch.sigmoid(x) * (x ** self.order)


Swish0 = nn.Sigmoid
Swish1 = partial_class(SwishN, order=1)
Swish2 = partial_class(SwishN, order=2)
Swish3 = partial_class(SwishN, order=3)
Swish4 = partial_class(SwishN, order=4)
Swish5 = partial_class(SwishN, order=5)
Swish = Swish1


class ResBlock(nn.Module):
    def __init__(self, n_neurons, act1=None, act2=None):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = act1 or Swish()
        self.lin2 = nn.Linear(n_neurons, n_neurons)
        self.act2 = act2 or Swish()

    def forward(self, x):
        y = x
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.acv2(x + y)


def get_resnet(n_input_units, n_output_units, n_res_blocks, n_res_units, actv_class=None):
    if actv_class is None:
        actv_class = Swish
    layers = []
    layers.append(nn.Linear(n_input_units, n_res_units))
    layers.append(actv_class())
    for i in range(n_res_blocks):
        layers.append(ResBlock(n_res_units, act1=actv_class(), act2=actv_class()))
    layers.append(nn.Linear(n_res_units, n_output_units))
    return nn.Sequential(*layers)

