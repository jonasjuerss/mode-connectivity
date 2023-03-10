from __future__ import annotations  # Allow using class type in class definition

import abc
from typing import Union, Tuple, List

import imageio.v2 as imageio
from argparse import Namespace

import imageio
import torch
from torch import Tensor

from jonas.coordinate_networks import CoordinateModule
from jonas.landscape_module import LandscapeModule

device = torch.device("cuda")


class TargetFunction(abc.ABC):
    def __init__(self, name: str, requested_coordinates: Union[Tensor, None]):
        self.name = name
        self.requested_coordinates = requested_coordinates

    @abc.abstractmethod
    def initialize(self, args: Namespace) -> TargetFunction:
        pass

    @abc.abstractmethod
    def evaluate(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                 corresponding_coords: Tensor, module: LandscapeModule, clipoff: Tensor = None)\
            -> Tuple[Tensor, Tensor]:
        """
        For predictions(_origin) and labels, it is expected that the first batch_size entries
        correspond to the first coordinate in corresponding_coords, the next to the second and so on.

        Both prediction tensors contain unnormalized logits.
        :param inputs: [batch_size * num_coords, *input_dims]
        :param predictions: [batch_size * num_coords, num_classes]
        :param labels: [batch_size * num_coords]
        :param prediction_losses: [batch_size * num_coords]
        :param corresponding_coords: [num_coords, num_dimensions]
        :param clipoff: An optional clipoff that the concrete function implements if it is not bound in one direction
        (e.g. crossentropy can get arbitrarily large). None by default which means that no clipoff will be applied. This
        is useful when using the target function for plots and not actual loss calculations.
        (Scalar GPU tensor for efficiency, So we don't have to load it every time)
        :return: (diversities, loss) where diversities is a [num_coords] tensor with the diversity for each coordinate
        and loss is the overall loss
        """
        pass


class PixelDifference2D(TargetFunction, abc.ABC):
    def __init__(self, name: str):
        super().__init__(name, None)
        self.target = None

    def initialize(self, args: Namespace) -> PixelDifference2D:
        # so target will have values like [v(0, 0), ..., v(0, n), v(1,0), ... , v(1, n), ...] where v(x1, x2) = -1 for
        # black pixels and 0 otherwise
        self.target = torch.Tensor(imageio.imread(f'res/icons/{args.target_image}.png') // 255)
        assert len(self.target.shape) <= 3
        if len(self.target.shape) == 3:
            self.target = torch.mean(self.target, dim=2)
        self.target_shape = self.target.shape
        if args.equal_weight_colors:
            white_percentage = self.target.sum() / (self.target_shape[0] * self.target_shape[1])
            self.target = self.target * ((0.5 / white_percentage) + (0.5 / (1 - white_percentage))) - (
                        0.5 / (1 - white_percentage))
            # self.target[self.target == 1] = 0.5 / white_percentage
            # self.target[self.target == 0] = -0.5 / (1 - white_percentage)
        else:
            self.target = 2 * (self.target - 0.5)
        self.target = self.target.reshape(-1).cuda()
        # can be expanded to more dimensions by just adding more linspaces and increasing the second number in reshape
        grid = torch.meshgrid(torch.linspace(0, 1, steps=self.target_shape[0]),
                              torch.linspace(0, 1, steps=self.target_shape[1]))
        self.requested_coordinates = torch.stack(grid).T.reshape(-1, 2).cuda()
        return self

    @abc.abstractmethod
    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        """
        The first batch_size entries correspond to the first coordinate in corresponding_coords,
        the next to the second and so on
        :param predictions: [batch_size * num_coords]
        :param prediction_losses: [batch_size * num_coords]
        :return: [batch_size * num_coords]
        """
        pass

    def evaluate(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                 corresponding_coords: Tensor, module: LandscapeModule, clipoff: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        White pixels mean high diversity. Therefore, (0, 0) == white would be unfeasible
        """

        # convert coords in [0, 1] to [0, height] and [0, width]
        # round just in case numerical issues give us e.g. 4.99 instead of 5 which would become 4 if casted to int
        corresponding_coords = torch.round(torch.stack((corresponding_coords[:, 0] * (self.target_shape[0] - 1),
                                                        corresponding_coords[:, 1] * (self.target_shape[1] - 1)),
                                                       dim=1)).to(int)
        # convert to coordinates in flattened vector
        # TODO I guess, now there is no need to flatten anymore in the first place
        corresponding_targets = self.target[
            corresponding_coords[:, 0] * self.target_shape[1] + corresponding_coords[:, 1]]

        diversities = self.measure_loss(inputs, predictions, labels, prediction_losses, corresponding_coords, module,
                                        clipoff=clipoff)
        # grouped mean for each coordinate. Only necessary because we want to return the diversity
        diversities = torch.mean(diversities.reshape(corresponding_coords.shape[0], -1), dim=1)
        return diversities, torch.mean(diversities * corresponding_targets)


class Euclidean2D(PixelDifference2D):
    def __init__(self):
        super().__init__("Euclidean2D")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        """
        :return: The euclidean distance between the predicted probabilities at the origin and the predicted
        probabilities at the given position
        """
        if compared_coords is None:
            compared_coords = torch.zeros((1, corresponding_coords.shape[-1]), device=device)

        # Convert unnormalized logits into actual probabilities
        predictions = torch.softmax(predictions, dim=-1)
        predictions_origin = torch.softmax(module(inputs, compared_coords), dim=-1)
        # Constant scaling factor so the result is roughly of magnitude 1 in the beginning. Note this means we can get a loss close to 10^6 later on
        return 5e2 * torch.mean(torch.square(predictions - predictions_origin), dim=-1)


class CrossEntropy2D(PixelDifference2D):
    def __init__(self):
        super().__init__("CrossEntropy2D")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        """
        :return: The cross entropy between the predicted probabilities at the origin and the predicted
        probabilities at the given position
        """
        # Whereas log probabilities are expected for the input,
        if compared_coords is None:
            compared_coords = torch.zeros((1, corresponding_coords.shape[-1]), device=device)

        predictions_origin = torch.softmax(module(inputs, compared_coords), dim=-1)
        res = torch.nn.functional.cross_entropy(predictions, predictions_origin, reduction='none')
        if clipoff is None:
            return res
        return torch.minimum(res, clipoff)

class ClassificationDifference(PixelDifference2D):
    def __init__(self):
        super().__init__("ClassificationDifference")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        """
        :return: The cross entropy between the predicted probabilities at the origin and the predicted
        probabilities at the given position
        """
        # Whereas log probabilities are expected for the input,
        if compared_coords is None:
            compared_coords = torch.zeros((1, corresponding_coords.shape[-1]), device=device)

        predictions_origin = module(inputs, compared_coords)
        return (torch.argmax(predictions_origin, dim=-1) != torch.argmax(predictions, dim=-1)).to(float)

class ClassificationErrorDifference(PixelDifference2D):
    def __init__(self):
        super().__init__("ClassificationErrorDifference")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        """
        :return: The cross entropy between the predicted probabilities at the origin and the predicted
        probabilities at the given position
        """
        # Whereas log probabilities are expected for the input,
        if compared_coords is None:
            compared_coords = torch.zeros((1, corresponding_coords.shape[-1]), device=device)

        labels_origin = torch.argmax(module(inputs, compared_coords), dim=-1)
        labels_here = torch.argmax(predictions, dim=-1)
        differing_entries = labels_origin != labels_here
        # equivalent to (but more efficient than) torch.logical_or(labels_origin != labels, labels_here != labels)
        erroneous_predictions = torch.logical_or(labels_origin != labels, differing_entries)
        # The constant factor in the end accounts for the fact that the parent class will just take the average over all
        # points instead of just the relevant ones (with erroneous prediction)
        return differing_entries.to(float) * (torch.numel(labels_origin) / torch.sum(erroneous_predictions))


class Loss2D(PixelDifference2D):

    def __init__(self):
        super().__init__("Loss2D")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        if clipoff is None:
            return prediction_losses
        return torch.minimum(prediction_losses, clipoff)

class NoTarget(PixelDifference2D):
    def __init__(self):
        super().__init__("NoTarget")

    def measure_loss(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                     corresponding_coords: Tensor, module: LandscapeModule, compared_coords: Tensor = None,
                     clipoff: Tensor = None):
        return -torch.ones_like(prediction_losses).cuda()

    def evaluate(self, inputs: Tensor, predictions: Tensor, labels: Tensor, prediction_losses: Tensor,
                 corresponding_coords: Tensor, module: LandscapeModule, clipoff: Tensor = None)\
            -> Tuple[Tensor, Tensor]:
        # Obviously, I should rather pass the device directly during creation but efficiency is less crucial here
        return -torch.ones(corresponding_coords.shape[0]).cuda(),\
            torch.tensor(-1).cuda()


__all__: List[TargetFunction] = [Euclidean2D(), CrossEntropy2D(), Loss2D(), ClassificationDifference(),
                                 ClassificationErrorDifference(), NoTarget()]


def function_from_name(name: str, args: Namespace) -> TargetFunction:
    for m in __all__:
        if m.name == name:
            return m.initialize(args)
    raise ValueError("Unknown target function:" + name)
