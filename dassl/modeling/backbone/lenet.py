import torch.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class LeNetBase(Backbone):

    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self._out_features = 50 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


@BACKBONE_REGISTRY.register()
def lenet(**kwargs):
    """
    This architecture was used for the USPS <-> MNIST task.
    """
    return LeNetBase()
