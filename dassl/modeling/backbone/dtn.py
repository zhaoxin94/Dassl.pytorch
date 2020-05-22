import torch.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class DTNBase(Backbone):

    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.Dropout2d(0.1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128), nn.Dropout2d(0.3), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU()
        )
        self._out_features = 256 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


@BACKBONE_REGISTRY.register()
def dtn(**kwargs):
    """
    This architecture was used for the SVHN -> MNIST task.
    """
    return DTNBase()
