import torch
import torch.nn as nn
from var_length_seq import VariableLengthConv1d

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blocks = []
        self.blocks.append(VariableLengthConv1d(1, 2, 3, 1, 0))
        self.blocks.append(VariableLengthConv1d(2, 4, 5, 2, 2))
        self.blocks.append(VariableLengthConv1d(4, 8, 1, 1, 0))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x