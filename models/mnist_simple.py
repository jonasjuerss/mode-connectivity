__all__ = ['MNISTSimple']

import math

from torch import nn


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
            nn.Linear(1600, num_classes),
            nn.Softmax()
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


class MNISTSimple:
    base = MNISTSimpleBase
    curve = None
    kwargs = {}
