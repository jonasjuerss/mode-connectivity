import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import List, cast

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
from jonas.landscape_module import LandscapeModule
from jonas.target_functions import TargetFunction, PixelDifference2D
from wandb_utils import log


def train_test(train_loader, model: LandscapeModule, optimizer, landscape_criterion: TargetFunction,
               plot_functions: List[PixelDifference2D], multi_plot_functions: List[PixelDifference2D],
               multi_plot_iterations, accuracy_weight, regularizer=None, lr_schedule=None, coordinates=None, train=True):
    loss_sum = 0.0
    landscape_loss_sum = 0.0
    prediction_loss_sum = 0.0
    accuracy = 0.0
    image_data = torch.zeros((coordinates.shape[0], 5 + len(plot_functions)))
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
                                                                                model)

                loss = accuracy_weight * prediction_loss + (1 - accuracy_weight) * landscape_loss
                assert not torch.isnan(loss).item()

                pred = output.data.argmax(1, keepdim=True)
                acc = 100 * pred.eq(target.data.view_as(pred)).to(float).mean().item()

                image_data[i, 0] += landscape_metric.item()
                image_data[i, 1] += landscape_loss.item()
                image_data[i, 2] += acc
                image_data[i, 3] += prediction_loss.item()
                image_data[i, 4] += loss.item()

                if plot_functions or multi_plot_functions: # Only enter context if actually necessary
                    with torch.no_grad() if train else nullcontext():
                        for j, fun in enumerate(plot_functions):
                            # Here, we know that we only feed information over a single coordinate so we can just
                            # take the overall mean as opposed to the per-coordinate mean calculated in the evaluate()
                            # method
                            image_data[i, 5 + j] += torch.mean(fun.measure_loss(input, output, target,
                                                                                prediction_loss.reshape(1, 1), coords,
                                                                                model)).item()
                        for j, fun in enumerate(multi_plot_functions):
                            multi_image_data[i, j] = torch.mean(fun.measure_loss(input, output, target,
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
                            multi_image_data[i + i_plot * coordinates.shape[0], j] = torch.mean(fun.measure_loss(input, output, target,
                                                                                 prediction_loss.reshape(1, 1),
                                                                                 coords, model,
                                                                                 max_diversity_positions[j])).item()




    image_data /= num_iters

    table = wandb.Table(columns=["x1", "x2", landscape_criterion.name, "landscape_loss", "accuracy", "prediction_loss",
                                 "loss"] + [f.name for f in plot_functions])
    for c in range(coordinates.shape[0]):
        table.add_data(coordinates[c, 0], coordinates[c, 1], *image_data[c])

    num_passes = len(train_loader.dataset)  # note that we already divide by coordinates.shape[0] in coordinate_scale
    res = {
        'loss': loss_sum / num_passes,
        'landscape_loss': landscape_loss_sum / num_passes,
        'prediction_loss': prediction_loss_sum / num_passes,
        'accuracy': accuracy / num_iters,
        'image': table,
        'scaling_factor': model.scaling_factor.item()
    }

    if multi_plot_functions:
        multi_table = wandb.Table(columns=["x1", "x2", "iteration"] + [f.name for f in multi_plot_functions])
        multi_image_data /= num_iters
        for c in range(coordinates.shape[0]):
            for i_plot in range(multi_plot_iterations):
                multi_table.add_data(coordinates[c, 0], coordinates[c, 1], i_plot, *multi_image_data[i_plot * coordinates.shape[0] + c])
        res["multi_plots"] = multi_table
    return res


def main(args):
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.dataset_scale,
        args.use_test
    )

    architecture = getattr(models, args.model)
    model = LandscapeModule(architecture, num_classes, args.landscape_dimensions, args.orthonormal_base,
                            args.learn_scaling_factor)
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
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        print("CAUTION: if you used --data_scale < 1, the data subset was random and a different subset will be taken"
              "now. This means \"training\" plot in the begining are more like evaluation plots (depending on how"
              "small your scale was) and if you continue training, this is not equivalent to if you had not stopped the"
              " run!")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
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
                                   args.multi_plot_iterations, args.accuracy_weight, regularizer,
                                   coordinates=target_function.requested_coordinates)
            log_dict.update(dict(
                scaling_factor=train_res["scaling_factor"],

                loss_train=train_res['loss'],
                prediction_loss_train=train_res['prediction_loss'],
                landscape_loss_train=train_res['landscape_loss'],
                acc_train=train_res['accuracy'],
                image_train=train_res["image"]))
        if not has_bn:
            test_res = train_test(loaders['test'], model, optimizer, target_function, plot_functions,
                                  multi_plot_functions, args.multi_plot_iterations, args.accuracy_weight, regularizer,
                                  coordinates=target_function.requested_coordinates, train=False)
            log_dict.update(dict(
                scaling_factor=test_res["scaling_factor"],

                loss_test=test_res['loss'],  # nll
                prediction_loss_test=test_res['prediction_loss'],
                landscape_loss_test=test_res['landscape_loss'],
                acc_test=test_res['accuracy'],
                image_test=test_res["image"],
            ))
            if multi_plot_functions:
                log_dict["multi_plots"] = test_res["multi_plots"]

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
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
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')

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
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')

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
                        help='a factor in [0, 1] that allows to scale down the data used')

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

    parser.set_defaults(test_only=False)
    parser.add_argument('--test_only', action='store_true', dest='test_only',
                        help='Only runs test phase')

    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')

    args = parser.parse_args()
    args = wandb_utils.init_wandb(args)
    main(args)
