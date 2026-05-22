import torch
import torch.nn as nn

class EnsembleModel(nn.Module):

    def __init__(self, num_classes=2000):

        super(EnsembleModel, self).__init__()

        self.fc = nn.Linear(
            num_classes * 4,
            num_classes
        )

    def forward(
        self,
        joint,
        bone,
        joint_motion,
        bone_motion
    ):

        x = torch.cat(
            [
                joint,
                bone,
                joint_motion,
                bone_motion
            ],
            dim=1
        )

        x = self.fc(x)

        return x