__all__ = ['MNISTNet']

import math

from torch import nn
import curves

class MNISTNetBase(nn.Module):

    def __init__(self, num_classes):
        super(MNISTNetBase, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1600, num_classes),
        )

        # Initialize weights
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)

class MNISTNetCurve(nn.Module):
    def __init__(self, num_classes, fix_points):
        super(MNISTNetCurve, self).__init__()
        self.conv1 = curves.Conv2d(1, 32, kernel_size=3, fix_points = fix_points)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = curves.Conv2d(32, 64, kernel_size=3, fix_points = fix_points)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten1 = nn.Flatten()
        self.linear1 = curves.Linear(1600, num_classes, fix_points = fix_points)

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x, coeffs_t)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten1(x)
        x = self.linear1(x, coeffs_t)
        return x


class MNISTNet:
    base = MNISTNetBase
    curve = MNISTNetCurve
    kwargs = {}
