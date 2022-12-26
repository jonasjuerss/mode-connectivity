import abc
import string

import torch.nn
from torch import Tensor

"""
TODO Currently unused
"""

class DifferenceMeasure(abc.ABC):

    def __init__(self, name: str):
        self.name = name
    @abc.abstractmethod
    def evaluate(self, a: torch.nn.Module, b: torch.nn.Module, data) -> Tensor:
        pass

class ClassificationDifference(DifferenceMeasure):
    """
    Note that this can be used for evaluation but not as loss as it is not differentiable
    """
    def __init__(self):
        super().__init__("ClassificationDifference")

    def evaluate(self, a: torch.nn.Module, b: torch.nn.Module, data) -> Tensor:
        # expected shape of a(data) and b(data): [num_samples, num_classes]
        res_a = torch.argmax(a(data), dim=-1)  # [num_samples]
        res_b = torch.argmax(b(data), dim=-1)  # [num_samples]
        return torch.sum(res_a == res_b) / res_a.shape[0]

class SquaredProbabilityDistance(DifferenceMeasure):

    def __init__(self):
        super().__init__("SquaredProbabilityDistance")
    def evaluate(self, a: torch.nn.Module, b: torch.nn.Module, data) -> Tensor:
        # expected shape: [num_samples, num_classes]
        res_a = a(data)
        res_b = b(data)
        return torch.sum(torch.square(res_a - res_b)) / res_a.shape[0]

__all__ = [ClassificationDifference(), SquaredProbabilityDistance()]

def measure_from_name(name: string) -> DifferenceMeasure:
    for m in __all__:
        if m.name== name:
            return m
    raise ValueError("Unknown difference measure:" + name)
