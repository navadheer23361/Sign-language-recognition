import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_gcn import STGCNBlock

class Model(nn.Module):

    def __init__(
        self,
        num_class=2000,
        num_point=27,
        num_person=1,
        graph=None
    ):

        super(Model, self).__init__()

        self.data_bn = nn.BatchNorm1d(
            num_person * 3 * num_point
        )

        self.layer1 = STGCNBlock(3,64)

        self.layer2 = STGCNBlock(64,128)

        self.layer3 = STGCNBlock(128,256)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(256, num_class)

    def forward(self, x):

        N,C,T,V,M = x.size()

        x = x.permute(0,4,3,1,2).contiguous()

        x = x.view(N, M * V * C, T)

        x = self.data_bn(x)

        x = x.view(N, M, V, C, T)

        x = x.permute(0,1,3,4,2).contiguous()

        x = x.view(N * M, C, T, V)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x