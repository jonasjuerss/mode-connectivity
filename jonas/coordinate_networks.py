import abc

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CoordinateModule(nn.Module, abc.ABC):
    def __init__(self, *base_modules: nn.Module):
        super().__init__()
        self.base_modules = nn.ModuleList(base_modules)
        self.ndims = len(base_modules) - 1
        assert self.ndims > 0

    @abc.abstractmethod
    def _forward_at(self, input: Tensor, coords: Tensor) -> Tensor:
        pass

    def forward_at_origin(self, input: Tensor) -> Tensor:
        """
        Returns predictions when all coordinates are zero. This method can be overwritten to deal
        with this frequent special case more efficiently
        """
        return self.base_modules[0](input)

    def forward(self, input: Tensor, coords: Tensor) -> Tensor:
        """

        :param input:
        :param coords: expected shape [*input.shape, num_dimensions]
        :return:
        """
        return self._forward_at(input, coords)


class CoordParameterFree(CoordinateModule):
    def __init__(self, *base_modules: nn.Module):
        super().__init__(*base_modules)
        self.base_modules = nn.ModuleList([self.base_modules[0]])  # to save memory

    def _forward_at(self, input: Tensor, coords: Tensor) -> Tensor:
        return self.base_modules[0](input)


class CoordLinear(CoordinateModule):

    def __init__(self, *base_modules: nn.Linear):
        super().__init__(*base_modules)

    def _forward_at(self, input: Tensor, coords: Tensor) -> Tensor:
        return F.linear(input,
                        self.base_modules[0].weight + sum([coords[0, i] * self.base_modules[i + 1].weight for i in range(self.ndims)]),
                        self.base_modules[0].bias + sum([coords[0, i] * self.base_modules[i + 1].bias for i in range(self.ndims)]))



class CoordConv2D(CoordinateModule):

    def __init__(self, *base_modules: nn.Conv2d):
        super().__init__(*base_modules)

    def _forward_at(self, input: Tensor, coords: Tensor) -> Tensor:
        return self.base_modules[0]._conv_forward(input,
                                                  self.base_modules[0].weight +
                                                  sum([coords[0, i] * self.base_modules[i + 1].weight
                                                       for i in range(self.ndims)]),
                                                  self.base_modules[0].bias +
                                                  sum([coords[0, i] * self.base_modules[i + 1].bias
                                                       for i in range(self.ndims)]))


class CoordSequential(CoordinateModule):

    def __init__(self, *base_modules: nn.Sequential):
        super().__init__(*base_modules)
        self.converted_modules = nn.ModuleList()
        for i in range(len(base_modules[0])):
            cur_modules = [base_modules[j][i] for j in range(len(base_modules))]
            self.converted_modules.append(convert_to_coord_modules(*cur_modules))

    def _forward_at(self, input: Tensor, coords: Tensor) -> Tensor:
        # Note that these asserts should be in all modules but we only put them here for efficiency
        assert coords.shape[-1] == self.ndims
        assert torch.allclose(coords, coords[0, :][None, :])\
               and "Different coordinates at the same pass are currently not supported for efficiency"
        if torch.count_nonzero(coords) == 0:
            return self.forward_at_origin(input)
        for module in self.converted_modules:
            input = module(input, coords)
        return input

    def forward_at_origin(self, input: Tensor) -> Tensor:
        for module in self.converted_modules:
            input = module.forward_at_origin(input)
        return input


def convert_to_coord_modules(*modules: nn.Module) -> CoordinateModule:
    if isinstance(modules[0], nn.Linear):
        module_type = CoordLinear
    elif isinstance(modules[0], nn.Conv2d):
        module_type = CoordConv2D
    elif isinstance(modules[0], nn.ReLU)\
            or isinstance(modules[0], nn.SELU)\
            or isinstance(modules[0], nn.Tanh)\
            or isinstance(modules[0], nn.Sigmoid)\
            or isinstance(modules[0], nn.Softmax)\
            or isinstance(modules[0], nn.Dropout)\
            or isinstance(modules[0], nn.MaxPool2d)\
            or isinstance(modules[0], nn.Flatten):
        module_type = CoordParameterFree
    elif isinstance(modules[0], nn.Sequential):
        module_type = CoordSequential
    else:
        raise NotImplementedError("Module type not supported yet:" + str(modules[0]))
    return module_type(*modules)
