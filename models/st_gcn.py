import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(STGCNBlock, self).__init__()

        self.gcn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9,1),
            padding=(4,0)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.gcn(x)

        x = self.tcn(x)

        x = self.bn(x)

        return F.relu(x)