import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

class NoiseDataset(Dataset):
    def __init__(self, original):
        self._original = original
        self._permutation = np.random.permutation(len(original))
    def __len__(self):
        return len(self._original)
    def __getitem__(self, idx):
        image, _ = self._original[idx]
        _, label = self._original[self._permutation[idx]]
        return image, label

    @property
    def data(self):
        return self._original.data

    @property
    def targets(self):
        return torch.tensor([self.__getitem__(i)[1] for i in range(len(self))])

class GPUDataset(TensorDataset):
    def __init__(self, original, device):
        inps = original.data
        tgts = original.targets

        try:
            inps = torch.tensor(inps)
            tgts = torch.tensor(tgts)
        except ValueError:
            pass

        inps = inps.to(device)
        tgts = tgts.to(device)

        super().__init__(inps, tgts)