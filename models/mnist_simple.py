__all__ = ['MNISTSimple', 'MNISTSimpleBase', 'MNISTFC']

import math

from torch import nn

import curves


class MNISTSimpleBase(nn.Sequential):

    def __init__(self, num_classes):
        super(MNISTSimpleBase, self).__init__(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            # TODO If necessary, get rid of this part as non-deterministic behaviour might make my landscape harder to achieve
            # nn.Dropout(0.5),
            nn.Linear(1600, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


class MNISTSimpleCurve(nn.Module):

    def __init__(self, num_classes, fix_points):
        super(MNISTSimpleCurve, self).__init__()

        self.conv1 = curves.Conv2d(1, 32, kernel_size=3, fix_points=fix_points)
        self.relu1 = nn.ReLU(True)
        self.mp1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = curves.Conv2d(32, 64, kernel_size=3, fix_points=fix_points)
        self.relu2 = nn.ReLU(True)
        self.mp2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()
        self.fc = curves.Linear(1600, num_classes, fix_points=fix_points)
        self.soft = nn.Softmax()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, coeffs_t):

        x = self.conv1(x, coeffs_t)
        x = self.relu1(x)
        x = self.mp1(x)

        x = self.conv2(x, coeffs_t)
        x = self.relu2(x)
        x = self.mp2(x)

        x = self.flat(x)
        x = self.fc(x, coeffs_t)
        x = self.soft(x)

        return x

class MNISTFCBase(nn.Sequential):

    def __init__(self, num_classes):
        super(MNISTFCBase, self).__init__(
            nn.Flatten(),
            nn.Linear(784, 800),
            nn.ReLU(True),
            nn.Linear(800, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()



class MNISTSimple:
    base = MNISTSimpleBase
    curve = MNISTSimpleCurve
    kwargs = {}

class MNISTFC:
    base = MNISTFCBase
    curve = None
    kwargs = {}
