import abc
from typing import List

import torch.nn
from torch.nn import Module, Parameter

from jonas.coordinate_networks import convert_to_coord_modules


class LandscapeModule(Module):

    def __init__(self, architecture: Module, num_classes: int, num_dimensions: int, orthonormal_base: bool,
                 learn_scaling_factor: bool, initial_scale: float, modules: List[Module] = None):
        """

        :param architecture: Architecture for the base networks
        :param num_classes: number of classes in the dataset
        :param num_dimensions: number of dimensions of the image
        :param orthonormal_base: Whether to enforce the base to be orthonormal
        :param learn_scaling_factor: Whether to multiply all coordinates by a learned factor
        :param initial_scale: the initial value of the learned factor
        :param modules: an optional list of modules to use as base points. By default, new modules will be initialized
        """
        super().__init__()
        self.orthonormal_base = orthonormal_base
        self.scaling_factor = torch.tensor(initial_scale, dtype=torch.float)
        self.num_classes = num_classes
        if learn_scaling_factor:
            self.scaling_factor = Parameter(self.scaling_factor)

        if modules is None:
            # Note: one might argue that subtract_origin=True would make sense here as well but both sre sensible
            # choices, the difference shouldn't be significant and I didn't want to make changes in the middle of the
            # experimentation phase
            self.coord_network = convert_to_coord_modules(False, *[architecture.base(num_classes=num_classes,
                                                                                     **architecture.kwargs)
                                                                   for _ in range(num_dimensions + 1)])
        else:
            assert num_dimensions + 1 == len(modules)
            self.coord_network = convert_to_coord_modules(True, *modules)

    def forward(self, data, coords):
        """
        :param data: [batch_size, feature_size] (in our case of image prediction: [batch_size, image_width, image_height])
        :param coords: [batch_size, num_dimensions]
        :return: [batch_size, num_classes]
        """
        coords = self.scaling_factor * coords
        if self.orthonormal_base:
            raise NotImplementedError()
        else:
            return self.coord_network(data, coords)