"""
    ResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import math
import torch.nn as nn

import curves

__all__ = ['ResNet110', 'ResNet164', 'ResNet20', 'ResNet20MNIST']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BasicBlockCurve, self).__init__()
        self.conv1 = conv3x3curve(inplanes, planes, stride=stride, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3curve(planes, planes, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)

        if self.downsample is not None:
            residual = self.downsample[0](x, coeffs_t)
            residual = self.downsample[1](residual, coeffs_t)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BottleneckCurve, self).__init__()
        self.conv1 = curves.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, planes * 4, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)

        out = self.conv3(out, coeffs_t)
        out = self.bn3(out, coeffs_t)

        if self.downsample is not None:
            residual = self.downsample[0](x, coeffs_t)
            residual = self.downsample[1](residual, coeffs_t)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):

    def __init__(self, num_classes, depth=110, avg_pool=8):
        super(ResNetBase, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock

        #self.inplanes = 16
        self.inplanes = 64
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.layer1 = self._make_layer(block, 16, n)
        self.layer1 = self._make_layer(block, 64, n)
        #self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer2 = self._make_layer(block, 128, n, stride=2)
        #self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer3 = self._make_layer(block, 256, n, stride=2)
        self.layer4 = self._make_layer(block, 512, n, stride=2)
        #self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(avg_pool)
        #self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)  # 4x4

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=110, avg_pool=8):
        super(ResNetCurve, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = BottleneckCurve
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlockCurve

        #self.inplanes = 16
        self.inplanes = 64
        #self.conv1 = curves.Conv2d(3, 16, kernel_size=3, padding=1,
        self.conv1 = curves.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False, fix_points=fix_points)
        self.bn = curves.BatchNorm2d(64, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        #self.layer1 = self._make_layer(block, 16, n, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 64, n, fix_points=fix_points)
        #self.layer2 = self._make_layer(block, 32, n, stride=2, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 128, n, stride=2, fix_points=fix_points)
        #self.layer3 = self._make_layer(block, 64, n, stride=2, fix_points=fix_points)
        self.layer3 = self._make_layer(block, 256, n, stride=2, fix_points=fix_points)
        self.layer4 = self._make_layer(block, 512, n, stride=2, fix_points=fix_points)
        #self.bn = curves.BatchNorm2d(64 * block.expansion, fix_points=fix_points)
        self.avgpool = nn.AvgPool2d(avg_pool)
        #self.fc = curves.Linear(64 * block.expansion, num_classes, fix_points=fix_points)
        self.fc = curves.Linear(512 * block.expansion, num_classes, fix_points=fix_points)

        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

    def _make_layer(self, block, planes, blocks, fix_points, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ModuleList([curves.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                       stride=stride, bias=False, fix_points=fix_points),
                        curves.BatchNorm2d(planes * block.expansion, fix_points=fix_points)])

        layers = list()
        layers.append(block(self.inplanes, planes, fix_points=fix_points, stride=stride,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fix_points=fix_points))

        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.bn(x, coeffs_t)
        x = self.relu(x)

        for block in self.layer1:  # 32x32
            x = block(x, coeffs_t)
        for block in self.layer2:  # 16x16
            x = block(x, coeffs_t)
        for block in self.layer3:  # 8x8
            x = block(x, coeffs_t)
        for block in self.layer4:  # 4x4
            x = block(x, coeffs_t)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)

        return x


class ResNet110:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 110}


class ResNet164:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 164}





# Lightweight
class ResNet20:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 20}

class ResNet20MNIST:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 20, 'avg_pool': 7}

class ResNet14:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 14}

class ResNet14MNIST:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 14, 'avg_pool': 7}

class ResNet8:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 8}

class ResNet8MNIST:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 8, 'avg_pool': 7}

class ResNet18:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 14, 'avg_pool': 4}