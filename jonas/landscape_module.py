import abc

import torch.nn
from torch.nn import Module, Parameter

from jonas.coordinate_networks import convert_to_coord_modules


class LandscapeModule(Module):

    def __init__(self, architecture: Module, num_classes: int, num_dimensions: int, orthonormal_base: bool, learn_scaling_factor: bool):
        """

        :param architecture: Architecture for the base networks
        :param num_classes: number of classes in the dataset
        :param num_dimensions: number of dimensions of the image
        :param orthonormal_base: Whether to enforce the base to be orthonormal
        :param learn_scaling_factor:
        """
        super().__init__()
        self.orthonormal_base = orthonormal_base
        self.scaling_factor = torch.tensor(1.0)
        if learn_scaling_factor:
            self.scaling_factor = Parameter(self.scaling_factor)

        self.coord_network = convert_to_coord_modules(*[architecture.base(num_classes=num_classes,
                                                                          **architecture.kwargs)
                                                        for _ in range(num_dimensions + 1)])

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