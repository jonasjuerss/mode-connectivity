import pickle
from typing import List

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from models import MNISTSimpleBase
from torch_autoneb.config import EvalConfig
from torch_autoneb.datasets import load_dataset
from torch_autoneb.models import DataModel, ModelWrapper, CompareModel


def get_draxler(i):
    with open(f'neb/graph{i}.pkl', 'rb') as file:
        data = pickle.load(file)
    edge = data.get_edge_data(1, 2, 4)
    return edge['path_coords'].numpy()


def get_garipov():
    paths_garipov = np.load('../garipov.npz')['X']
    return paths_garipov[0, ...]

def evaluate(params: np.ndarray):
    base_model = MNISTSimpleBase(10)
    model = ModelWrapper(DataModel(CompareModel(base_model, CrossEntropyLoss()), load_dataset('mnist')[0]))
    model.adapt_to_config(EvalConfig(128))
    model.set_coords_no_grad(torch.Tensor(params))
    loss = model.apply(gradient=False)
    return loss


def path_length(path: np.ndarray) -> np.ndarray:
    return np.sum([np.linalg.norm(path[i]-path[i+1]) for i in range(path.shape[0] - 1)])


def sample_from_path_uniform(path: np.ndarray, t: float) -> np.ndarray:
    assert 0 <= t <= 1

    t = t * path_length(path)
    for i in range(path.shape[0] - 1):
        dist = np.linalg.norm(path[i]-path[i+1])
        if dist > t:
            # Found segment from which to return
            out = path[i] + (path[i+1]-path[i])*t/dist
            return out
        else:
            # Keep iterating
            t = t - dist
    return path[-1]


def evaluate_along_path(path: np.ndarray, steps: int) -> np.ndarray:
    out = np.array([evaluate(sample_from_path_uniform(path, i / steps)) for i in range(steps + 1)])
    return out


def sample_interpolated(paths: List[np.ndarray], t: float, w: np.ndarray = None) -> np.ndarray:
    coeffs = np.array([sample_from_path_uniform(p, t) for p in paths])
    average = np.average(coeffs, weights=w, axis=0)
    return average


def evaluate_interpolated(paths: List[np.ndarray], steps: int, w: np.ndarray = None) -> np.ndarray:
    out = np.array([evaluate(sample_interpolated(paths, i / steps, w)) for i in range(steps + 1)])
    return out


def main():
    STEPS = 200

    paths = [get_draxler(i) for i in range(5)]
    paths.append(get_garipov())

    oversampled = np.array([[sample_from_path_uniform(p, t) for t in np.linspace(0, 1, 1000, endpoint=True)]
                            for p in  paths])
    oversampled = oversampled.reshape(len(paths) * 1000, -1)
    np.savez('../oversampled.npz', X=oversampled)

    # interp = evaluate_interpolated(paths, steps=STEPS, w=None)
    # draxlers = np.array([evaluate_along_path(p, steps=STEPS) for p in paths])

    # total_mean = evaluate_interpolated(paths, steps=STEPS, w=np.array([1, 1, 1, 1, 1, 5]))
    # print(total_mean.max())
    # with open('neb/total.pkl', 'wb') as file:
    #     pickle.dump(total_mean, file)

    # with open('neb/draxlers.pkl', 'wb') as file:
    #     pickle.dump(draxlers, file)
    #
    # with open('neb/interpolated.pkl', 'wb') as file:
    #     pickle.dump(interp, file)
    #
    # linear = paths[0][[0, -1], ...]
    # barrier = evaluate_along_path(linear, steps=STEPS)
    #
    # with open('neb/barrier.pkl', 'wb') as file:
    #     pickle.dump(barrier, file)


if __name__ == "__main__":
    main()
