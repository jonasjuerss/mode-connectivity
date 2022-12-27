from __future__ import annotations # Allow using class type in class definition

import abc
from typing import Union, Tuple

import imageio.v2 as imageio
from argparse import Namespace

import imageio
import torch
from torch import Tensor


class TargetFunction(abc.ABC):
    def __init__(self, name: str, requested_coordinates: Union[Tensor, None]):
        self.name = name
        self.requested_coordinates = requested_coordinates

    @abc.abstractmethod
    def initialize(self, args: Namespace) -> TargetFunction:
        pass

    @abc.abstractmethod
    def evaluate(self, predictions_origin: Tensor, predictions: Tensor, labels: Tensor, corresponding_coords: Tensor) -> Tuple[Tensor, Tensor]:
        """
        For predictions(_origin) and labels, it is expected that the first batch_size entries
        correspond to the first coordinate in corresponding_coords, the next to the second and so on.

        Both prediction tensors contain unnormalized logits.
        :param predictions_origin: [batch_size * num_coords, num_classes]
        :param predictions: [batch_size * num_coords, num_classes]
        :param labels: [batch_size * num_coords]
        :param corresponding_coords: [num_coords, num_dimensions]
        :return: (diversities, loss) where diversities is a [num_coords] tensor with the diversity for each coordinate
        and loss is the overall loss
        """
        pass


class PixelDifference2D(TargetFunction):
    """
    TODO: possibly add an option for grey "don't care" pixels. That can make it more efficient
    """
    def __init__(self):
        super().__init__("PixelDifference2D", None)
        self.target = None

    def initialize(self, args: Namespace) -> PixelDifference2D:
        # so target will have values like [v(0, 0), ..., v(0, n), v(1,0), ... , v(1, n), ...] where v(x1, x2) = -1 for
        # black pixels and 0 otherwise
        self.target = torch.Tensor(imageio.imread(f'res/icons/{args.target_image}.png') / 255 - 1.)
        self.target_shape = self.target.shape
        self.target = self.target.reshape(-1).cuda()
        # can be expanded to more dimensions by just adding more linspaces and increasing the second number in reshape
        grid = torch.meshgrid(torch.linspace(0, 1, steps=self.target_shape[0]),
                              torch.linspace(0, 1, steps=self.target_shape[1]))
        self.requested_coordinates = torch.stack(grid).T.reshape(-1, 2).cuda()
        return self

    def evaluate(self, predictions_origin: Tensor, predictions: Tensor, labels: Tensor, corresponding_coords: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO note that currently, in contrast to the sightseeing paper, each pixel has the same weight instead of both
        #  colors having the same weight

        # Convert unnormalized logits into actual probabilities
        predictions = torch.softmax(predictions, dim=-1)
        predictions_origin = torch.softmax(predictions_origin, dim=-1)

        # convert coords in [0, 1] to [0, height] and [0, width]
        # round just in case numerical issues give us e.g. 4.99 instead of 5 which would become 4 if casted to int
        corresponding_coords = torch.round(torch.stack((corresponding_coords[:, 0] * (self.target_shape[0] - 1),
                                                        corresponding_coords[:, 1] * (self.target_shape[1] - 1)), dim=1)).to(int)
        # convert to coordinates in flattened vector
        # TODO I guess, now there is no need to flatten anymore in the first place
        corresponding_targets = self.target[corresponding_coords[:, 0] * self.target_shape[1] + corresponding_coords[:, 1]]

        # TODO try KL divergence instead of euclidean distance
        diversities = torch.mean(torch.square(predictions - predictions_origin), dim=-1)  # [batch_size * num_coords]
        # grouped mean for each coordinate. Only necessary because we want to return the diversity
        diversities = torch.mean(diversities.reshape(corresponding_coords.shape[0], -1), dim=1)
        return diversities, torch.mean(diversities * corresponding_targets)


__all__ = [PixelDifference2D()]


def function_from_name(name: str, args: Namespace) -> TargetFunction:
    for m in __all__:
        if m.name == name:
            return m.initialize(args)
    raise ValueError("Unknown target function:" + name)
