import argparse
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import List, cast, Union

import tabulate
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import data
import models
import utils
import wandb_utils
from jonas import target_functions
from jonas.landscape_module import LandscapeModule, MultiLandscapeModule
from jonas.target_functions import TargetFunction, PixelDifference2D
from wandb_utils import log


def train_test(train_loader, model: Union[LandscapeModule, MultiLandscapeModule], optimizer,
               landscape_criterion: TargetFunction, plot_functions: List[PixelDifference2D],
               multi_plot_functions: List[PixelDifference2D],  multi_plot_iterations, accuracy_weight, clipoff,
               regularizer=None, lr_schedule=None, coordinates=None, train=True):
    loss_sum = 0.0
    landscape_loss_sum = 0.0
    prediction_loss_sum = 0.0
    accuracy = 0.0
    image_data = torch.zeros((coordinates.shape[0], 5 + model.num_classes + len(plot_functions)))
    all_classes = torch.arange(model.num_classes)[None, :].cuda()
    if multi_plot_functions:
        multi_image_data = torch.zeros((multi_plot_iterations * coordinates.shape[0], len(multi_plot_functions)))

    # scale down (prediction) loss because it is accumulated over coordinates
    num_iters = len(train_loader)
    if train:
        model.train()
    else:
        model.eval()
    # train: 391 iterations, 12:39min for small architecture, test 79 iterations/00:52min
    with nullcontext() if train else torch.no_grad():
        for iter, (input, target) in tqdm(enumerate(train_loader), total=num_iters):
            if lr_schedule is not None:
                lr = lr_schedule(iter / num_iters)
                utils.adjust_learning_rate(optimizer, lr)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Process coordinates 1 by 1 to keep batch_size fixed.
            if train:
                optimizer.zero_grad()
            for i in range(coordinates.shape[0]):
                coords = (coordinates[i, :])[None, :]
                output = model(input, coords.expand(input.shape[0], -1))

                prediction_loss = F.cross_entropy(output, target)
                if regularizer is not None:
                    prediction_loss += regularizer(model)
                landscape_metric, landscape_loss = landscape_criterion.evaluate(input, output, target,
                                                                                prediction_loss.reshape(1, 1), coords,
                                                                                model, clipoff)

                loss = accuracy_weight * prediction_loss + (1 - accuracy_weight) * landscape_loss
                if torch.isnan(loss).item():
                    print(output)
                    print("coords: ", coords)
                    print("accuracy_weight:", accuracy_weight)
                    print("prediction_loss:", prediction_loss)
                    print("landscape_loss:", landscape_loss)
                    raise ValueError("NaN loss. Something went wrong with the training.")

                # [batch_size, 1]
                pred = output.data.argmax(1, keepdim=True)
                acc = 100 * pred.eq(target.data.view_as(pred)).to(float).mean().item()

                image_data[i, 0] += landscape_metric.item()
                image_data[i, 1] += landscape_loss.item()
                image_data[i, 2] += acc
                image_data[i, 3] += prediction_loss.item()
                image_data[i, 4] += loss.item()
                image_data[i, 5:5 + model.num_classes] += (100 * torch.sum(pred == all_classes, dim=0) / pred.shape[0]).cpu()

                if plot_functions or multi_plot_functions: # Only enter context if actually necessary
                    with torch.no_grad() if train else nullcontext():
                        for j, fun in enumerate(plot_functions):
                            # Here, we know that we only feed information over a single coordinate so we can just
                            # take the overall mean as opposed to the per-coordinate mean calculated in the evaluate()
                            # method
                            image_data[i, 5 + model.num_classes + j] += torch.mean(fun.measure_loss(input, output,
                                                                                                    target,
                                                                                                    prediction_loss\
                                                                                                    .reshape(1, 1),
                                                                                                    coords,
                                                                                                    model)).item()
                        for j, fun in enumerate(multi_plot_functions):
                            multi_image_data[i, j] += torch.mean(fun.measure_loss(input, output, target,
                                                                                 prediction_loss.reshape(1, 1),
                                                                                 coords, model)).item()
                # Important: don't just move that into loss_sum as it is also used in backward()
                loss = loss / coordinates.shape[0]
                prediction_loss = prediction_loss / coordinates.shape[0]
                landscape_loss = landscape_loss / coordinates.shape[0]

                loss_sum += loss.item() * input.size(0)
                prediction_loss_sum += prediction_loss.item() * input.size(0)
                landscape_loss_sum += landscape_loss.item() * input.size(0)
                accuracy += acc / coordinates.shape[0]

                if train:
                    # accumulate gradients over all coordinates before actually updating them together
                    # Memory usage would explode if we had to pass all of them at once
                    loss.backward()
            if train:
                optimizer.step()

            # Iterate over data batch again for creating the subsequent multi plots
        if multi_plot_functions:
            for i_plot in range(1, multi_plot_iterations):
                max_diversity_positions = []
                for j in range(0, len(multi_plot_functions)):
                    # shape: [num_coords]
                    min_diversity, _ = torch.min(multi_image_data[:, j].reshape(multi_plot_iterations,
                                                                                coordinates.shape[0])[:i_plot],
                                                 dim=0)
                    max_diversity_positions.append(coordinates[torch.argmax(min_diversity)][None, :])

                for iter, (input, target) in tqdm(enumerate(train_loader), total=num_iters):
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    # Process coordinates 1 by 1 to keep batch_size fixed.
                    for i in range(coordinates.shape[0]):
                        coords = (coordinates[i, :])[None, :]
                        output = model(input, coords.expand(input.shape[0], -1))
                        for j, fun in enumerate(multi_plot_functions):
                            multi_image_data[i + i_plot * coordinates.shape[0], j] += torch.mean(fun.measure_loss(input, output, target,
                                                                                 prediction_loss.reshape(1, 1),
                                                                                 coords, model,
                                                                                 max_diversity_positions[j])).item()




    image_data /= num_iters

    table = wandb.Table(columns=["x1", "x2", landscape_criterion.name, "landscape_loss", "accuracy", "prediction_loss",
                                 "loss"] + [f"Class_{i}" for i in range(10)] + [f.name for f in plot_functions])
    for c in range(coordinates.shape[0]):
        table.add_data(coordinates[c, 0], coordinates[c, 1], *image_data[c])

    num_passes = len(train_loader.dataset)  # note that we already divide by coordinates.shape[0] in coordinate_scale
    res = {
        'loss': loss_sum / num_passes,
        'landscape_loss': landscape_loss_sum / num_passes,
        'prediction_loss': prediction_loss_sum / num_passes,
        'accuracy': accuracy / num_iters,
        'image': table,
        'scaling_factor': model.scaling_factor.item() if hasattr(model, "scaling_factor") else -1
    }

    if multi_plot_functions:
        multi_table = wandb.Table(columns=["x1", "x2", "iteration"] + [f.name for f in multi_plot_functions])
        multi_image_data /= num_iters
        for c in range(coordinates.shape[0]):
            for i_plot in range(multi_plot_iterations):
                multi_table.add_data(coordinates[c, 0], coordinates[c, 1], i_plot,
                                     *multi_image_data[i_plot * coordinates.shape[0] + c])
        res["multi_plots"] = multi_table
    return res


def main(args):
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True

    # This way, we will always generate the same dataset
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.dataset_scale,
        args.use_test
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    architecture = getattr(models, args.model)
    base_modules = None
    if args.base_points:
        print(f'Using endpoints {args.base_points}')
        base_modules = [architecture.base(num_classes=num_classes, **architecture.kwargs)
                        for _ in args.base_points]
        for i, m in enumerate(base_modules):
            m.load_state_dict(torch.load(args.base_points[i])['model_state'])

    model = LandscapeModule(architecture, num_classes, args.landscape_dimensions, args.orthonormal_base,
                            args.learn_scaling_factor, args.initial_scale, base_modules)
    model.cuda()

    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr

    # difference_measure = measure_from_name(args.diversity_function)
    # criterion = difference_measure.evaluate
    target_function = target_functions.function_from_name(args.landscape_function, args)
    if args.landscape_function in args.plot_metrics:
        args.plot_metrics.remove(args.landscape_function)
    plot_functions = [target_functions.function_from_name(f, args) for f in args.plot_metrics]
    plot_functions = cast(List[PixelDifference2D], plot_functions)
    multi_plot_functions = [target_functions.function_from_name(f, args) for f in args.multi_plot_metrics]
    multi_plot_functions = cast(List[PixelDifference2D], multi_plot_functions)
    regularizer = None
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    start_epoch = 1
    if len(args.resume) == 1:
        print('Resume training from %s' % args.resume[0])
        checkpoint = torch.load(args.resume[0])
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    elif len(args.resume) > 1:
        landscape_modules = []
        for c in args.resume:
            m = LandscapeModule(architecture, num_classes, args.landscape_dimensions, args.orthonormal_base,
                                args.learn_scaling_factor, args.initial_scale, base_modules)
            m.cuda()
            m.load_state_dict(torch.load(c)['model_state'])
            landscape_modules.append(m)
        model = MultiLandscapeModule(*landscape_modules)
        model.cuda()


    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        use_wandb=args.wandb_checkpoints,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    clipoff = torch.tensor(args.loss_clipoff).cuda()

    coords_per_module = target_function.requested_coordinates
    requested_coordinates = coords_per_module.repeat(max(1, len(args.resume)), 1)
    requested_coordinates[:, 1] += (torch.arange(max(1, len(args.resume))) * 2).repeat_interleave(coords_per_module.shape[0]).cuda()
    for epoch in range(start_epoch, (start_epoch if args.test_only else args.epochs) + 1):
        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)

        log_dict = dict(
            epoch=epoch,
            learning_rate=lr
        )

        if not args.test_only:
            train_res = train_test(loaders['train'], model, optimizer, target_function, plot_functions, [],
                                   args.multi_plot_iterations, args.accuracy_weight, clipoff, regularizer,
                                   coordinates=requested_coordinates)
            log_dict.update(dict(
                scaling_factor=train_res["scaling_factor"],

                loss_train=train_res['loss'],
                prediction_loss_train=train_res['prediction_loss'],
                landscape_loss_train=train_res['landscape_loss'],
                acc_train=train_res['accuracy'],
                image_train=train_res["image"]))
        if not has_bn:
            test_res = train_test(loaders['test'], model, optimizer, target_function, plot_functions,
                                  multi_plot_functions if epoch % args.multi_plot_freq == 1 else [],
                                  args.multi_plot_iterations, args.accuracy_weight, clipoff, regularizer,
                                  coordinates=requested_coordinates, train=False)
            log_dict.update(dict(
                scaling_factor=test_res["scaling_factor"],

                loss_test=test_res['loss'],  # nll
                prediction_loss_test=test_res['prediction_loss'],
                landscape_loss_test=test_res['landscape_loss'],
                acc_test=test_res['accuracy'],
                image_test=test_res["image"],
            ))
            if "multi_plots" in test_res:
                log_dict["multi_plots"] = test_res["multi_plots"]

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                use_wandb=args.wandb_checkpoints,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        log_dict["epoch_duration"] = time.time() - time_ep
        log(log_dict)

        # values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['loss'],
        #           test_res['accuracy'], time_ep]
        # table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        # if epoch % 40 == 1 or epoch == start_epoch:
        #     table = table.split('\n')
        #     table = '\n'.join([table[1]] + table)
        # else:
        #     table = table.split('\n')[2]
        # print(table)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            use_wandb=args.wandb_checkpoints,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default=os.path.join("networks", "auto",
                                                                datetime.now().strftime("%d-%m-%Y_%H-%M-%S")),
                        metavar='DIR', help='training directory (default: networks/auto/<current date ad time>)')

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')

    parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--resume', nargs='+', default=[], metavar='CKPT',
                        help='Checkpoint(s) to resume training from (default: None). Multiple can be given to evaluate '
                             'multiple coordinate systems together.')
    parser.add_argument('--base_points', nargs='+', default=[],
                        help='Gives multiple checkpoints of separate modules to use as base points for the coordinate '
                             'system. The origin will be subtracted from the other ones so the edges of the coordinate'
                             ' system will actually be the networks. Make sure all of them follow the same architecture.')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--dataset_scale', type=float, default=1,
                        help='A factor in [0, 1] that allows to scale down the data used. This will create a random '
                             'subset with seed 1 (independent of --seed argument) such that the underlying dataset will'
                             ' always be the same fo the same scale.')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--landscape_dimensions', type=int, default=2,
                        help='The number of dimensions in the landscape we\'re searching for. '
                             'E.g. 2 for a "normal" image')
    parser.set_defaults(orthonormal_base=False)
    parser.add_argument('--orthonormal_base', action='store_true', dest='orthonormal_base',
                        help='Whether the base of the landscape should be forced to be orthonormal. Currently not supported.')
    parser.set_defaults(learn_scaling_factor=True)
    parser.add_argument('--no_scaling_factor', action='store_false', dest='learn_scaling_factor',
                        help='Whether to learn an additional scaling factor for the image like the loss landscape '
                             'sightseeing paper does for their orthonormal base.')
    parser.set_defaults(equal_weight_colors=False)
    parser.add_argument('--equal_weight_colors', action='store_true', dest='equal_weight_colors',
                        help='Whether the two colors (black and white) in target images should have the same weight like'
                             'in the sightseeing paper. By default, each pixel has the same weight which might lead to a'
                             'preference of just going into the loss/diversity direction that matches more pixels.')
    # parser.add_argument('--diversity_function', type=str, default="SquaredProbabilityDistance",
    #                     help='The difference function between networks to use as diversity measure')

    parser.add_argument('--landscape_function', type=str, default="CrossEntropy2D",
                        help='The function to optimize the diversity landscape for')
    parser.add_argument('--target_image', type=str, default=None,
                        help='The name of the target image in the icons folder')
    parser.add_argument('--accuracy_weight', type=float, default=0,
                        help='The weight of the actual prediction accuracy. By default, this is 0, so the network only'
                             'tries to attain the diversity image without improving the actual prediction accuracy')
    parser.add_argument('--plot_metrics', nargs='+', default=[],
                        help='Enforces plotting additional landscape metrics, even if they are not required for the landscape loss')
    parser.add_argument('--multi_plot_metrics', nargs='+', default=[],
                        help='Plot the given diversity metrics multi_plot_iterations times when testing, starting with'
                             'the origin and then using the point which is worst explained by the previous points in the'
                             ' following iterations.')
    parser.add_argument('--multi_plot_iterations', type=int, default=5,
                        help='How many plots to create for --multi_plot_metrics')
    parser.add_argument('--multi_plot_freq', type=int, default=5,
                        help='Frequency with which to log multi_plots')
    parser.add_argument('--loss_clipoff', type=float, default=2.5,
                        help='The clipoff for loss. Note that Crossentropy would have no upper bound.')
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='The initial scale for the coordinate system.')

    parser.set_defaults(test_only=False)
    parser.add_argument('--test_only', action='store_true', dest='test_only',
                        help='Only runs test phase')

    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')
    parser.set_defaults(wandb_checkpoints=False)
    parser.add_argument('--wandb_checkpoints', action='store_true', dest='wandb_checkpoints',
                        help='Stores checkpoints to weights and biases.')
    parser.add_argument('--continue_wandb', type=str, default=None,
                        help='If this is provided, all other arguments are ignored and a run is loaded from weights and'
                             'biases.')

    args = parser.parse_args()
    args.wandb_log = False  # To fix the issue Miran introduced by using his own argument name

    if args.continue_wandb is not None:
        raise NotImplementedError()

    args = wandb_utils.init_wandb(args)
    main(args)
