import copy
import os
from collections import OrderedDict

import torch
import pickle

from networkx import MultiGraph
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR

import utils
from models import MNISTSimpleBase
from torch_autoneb import neb, auto_neb, landscape_exploration, suggest
from torch_autoneb.config import NEBConfig, OptimConfig, EvalConfig, AutoNEBConfig, LandscapeExplorationConfig, \
    replace_instanciation
from torch_autoneb.datasets import load_dataset
from torch_autoneb.models import ModelWrapper, DataModel, CompareModel
from torch_autoneb.fill import equal, highest
from torch.optim import Adam


def from_dict(config_dict: dict):
    config_dict = copy.deepcopy(config_dict)
    config_dict["suggest_methods"], config_dict["suggest_args"] = zip(
        *[replace_instanciation(engine, suggest) for engine in config_dict["suggest"]])
    del config_dict["suggest"]
    return LandscapeExplorationConfig(**config_dict)


def load_model(path: str, num: int = 200) -> torch.nn.Module:
    state_dict = torch.load(f'checkpoint/{path}/checkpoint-{num}.pt')['model_state']
    model = MNISTSimpleBase(10)
    model.load_state_dict(state_dict)
    return model.cuda()


def main():
    EDPT_1 = "mnist/edpt1"
    EDPT_2 = "mnist/edpt2"
    NUM_POINTS = 200
    NUM_ITER = 10

    edpt_1 = ModelWrapper(load_model(EDPT_1))
    edpt_2 = ModelWrapper(load_model(EDPT_2))

    base_model = MNISTSimpleBase(10)
    model = ModelWrapper(DataModel(CompareModel(base_model, CrossEntropyLoss()), load_dataset('mnist')[0]))

    minima = [edpt_1.get_coords(), edpt_2.get_coords()]

    config = EvalConfig(1024)  # Set batch size
    neb_optim_config = OptimConfig(nsteps=NUM_ITER,
                                   algorithm_type=Adam,
                                   algorithm_args={"lr": 0.01},
                                   scheduler_type=LinearLR,
                                   scheduler_args={'start_factor': 1.0,
                                                   'end_factor': 0.01,
                                                   'total_iters': int(0.9 * NUM_ITER)},
                                   eval_config=config)
    neb_config = NEBConfig(spring_constant=10,
                           weight_decay=1e-4,
                           insert_method=equal,
                           insert_args={"count": NUM_POINTS - 2},
                           subsample_pivot_count=1,
                           optim_config=neb_optim_config)
    print('Starting NEB')
    out = neb({
        "path_coords": torch.cat([m.view(1, -1) for m in minima]),
        "target_distances": torch.ones(1)
        }, model, neb_config)

    models_out = [None] * NUM_POINTS
    new_keys = ['net.conv1.weight_', 'net.conv1.bias_',
                'net.conv2.weight_', 'net.conv2.bias_',
                'net.fc.weight_', 'net.fc.bias_']

    total_state_dict = OrderedDict({'coeff_layer.range': torch.tensor(list(range(NUM_POINTS)))})
    for idx, m in enumerate(models_out):
        model.set_coords_no_grad(out['path_coords'][idx])
        state_dict = copy.deepcopy(model.model.model.model.state_dict())
        state_dict = {f'{new_keys[j]}{idx}': v for j, v in enumerate(state_dict.values())}
        total_state_dict.update(state_dict)
    total_state_dict = OrderedDict(sorted(total_state_dict.items()))

    x = 'path_neb_1'
    dir = f'D:\Projects\PycharmProjects\mode-connectivity\checkpoint\mnist\{x}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    utils.save_checkpoint(
        dir,
        NUM_ITER,
        name='checkpoint',
        model_state=total_state_dict,
        )


def autoneb():
    EDPT_1 = "mnist/edpt1"
    EDPT_2 = "mnist/edpt2"
    NUM_POINTS = 3
    NUM_ITER = 200

    edpt_1 = ModelWrapper(load_model(EDPT_1))
    edpt_2 = ModelWrapper(load_model(EDPT_2))

    base_model = MNISTSimpleBase(10)
    model = ModelWrapper(DataModel(CompareModel(base_model, CrossEntropyLoss()), load_dataset('mnist')[0]))

    model.to('cuda:0')
    edpt_1.to('cuda:0')
    edpt_2.to('cuda:0')

    minima = [edpt_1.get_coords(), edpt_2.get_coords()]

    config = EvalConfig(1024)  # Set batch size
    neb_optim_config = OptimConfig(nsteps=NUM_ITER,
                                   algorithm_type=Adam,
                                   algorithm_args={"lr": 0.01},
                                   scheduler_type=LinearLR,
                                   scheduler_args={'start_factor': 1.0,
                                                   'end_factor': 0.01,
                                                   'total_iters': int(0.9 * NUM_ITER)},
                                   eval_config=config)
    neb_config = NEBConfig(spring_constant=float('inf'),
                           weight_decay=1e-4,
                           insert_method=equal,
                           insert_args={"count": NUM_POINTS},
                           subsample_pivot_count=1,
                           optim_config=neb_optim_config)

    neb_config2 = NEBConfig(spring_constant=float('inf'),
                            weight_decay=1e-4,
                            insert_method=highest,
                            insert_args={"count": NUM_POINTS, 'key': 'dense_train_loss'},
                            subsample_pivot_count=3,
                            optim_config=neb_optim_config)
    autoneb_cfg = AutoNEBConfig([neb_config, neb_config2, neb_config2, neb_config2])
    lex_cfg = from_dict({'value_key': 'train_loss',
                         'weight_key': 'saddle_train_loss',
                         'suggest': ['unfinished', 'disconnected', 'mst'],
                         'auto_neb_config': autoneb_cfg})
    graph = MultiGraph()
    for idx, minimum in enumerate(minima):
        graph.add_node(idx + 1, **{'coords': minimum,
                                   'dense_train_loss': 0,
                                   'train_loss': 0})
    print('Starting LEX')
    landscape_exploration(graph, model, lex_cfg)

    with open('neb/graph.pkl', 'wb') as file:
        pickle.dump(graph, file)


if __name__ == "__main__":
    autoneb()
