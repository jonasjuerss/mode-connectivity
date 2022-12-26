import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import wandb_utils
from jonas import target_functions
from jonas.difference_measures import measure_from_name
from jonas.landscape_module import LandscapeModule
from wandb_utils import log


def train_test(train_loader, model, optimizer, landscape_criterion, accuracy_weight, regularizer=None, lr_schedule=None, coordinates=None, train=True):
    loss_sum = 0.0
    landscape_loss_sum = 0.0
    prediction_loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    if train:
        model.train()
    else:
        model.eval()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            utils.adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Process coordinates 1 by 1 to keep batch_size fixed.
        if train:
            optimizer.zero_grad()
        for i in range(coordinates.shape[0]):
            origin, output = model(input, ((coordinates[i])[None, :]).expand(input.shape[0], -1))

            loss = F.cross_entropy(output, target)
            if regularizer is not None:
                loss += regularizer(model)
            prediction_loss_sum += loss.item() * input.size(0)
            landscape_loss = landscape_criterion(origin, output, target, (coordinates[i])[None, :])
            loss = accuracy_weight * loss + landscape_loss

            loss_sum += loss.item() * input.size(0)
            landscape_loss_sum += landscape_loss.item() * input.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            if train:
                # accumulate gradients over all coordinates before actually updating them together
                # Memory usage would explode if we had to pass all of them at once
                loss.backward()

        if train:
            optimizer.step()

    num_passes = len(train_loader.dataset) * coordinates.shape[0]
    return {
        'loss': loss_sum / num_passes,
        'landscape_loss': landscape_loss_sum / num_passes,
        'prediction_loss': prediction_loss_sum / num_passes,
        'accuracy': correct * 100.0 / num_passes,
    }

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
        args.use_test
    )

    architecture = getattr(models, args.model)
    model = LandscapeModule(architecture, num_classes, args.landscape_dimensions, args.orthonormal_base, args.learn_scaling_factor)
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
    #criterion = difference_measure.evaluate
    target_function = target_functions.function_from_name(args.landscape_function, args)
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
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)


        train_res = train_test(loaders['train'], model, optimizer, target_function.evaluate, args.accuracy_weight,
                                   regularizer, coordinates=target_function.requested_coordinates)
        if not has_bn:
            # TODO plot image
            test_res = train_test(loaders['test'], model, optimizer, target_function.evaluate, args.accuracy_weight,
                                  regularizer, coordinates=target_function.requested_coordinates)

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep

        log(dict(
            epoch=epoch,
            learning_rate=lr,

            loss_train=train_res['loss'],
            prediction_loss_train=train_res['prediction_loss'],
            landscape_loss_train=train_res['landscape_loss'],
            acc_train=train_res['accuracy'],

            loss_test=test_res['loss'],  #nll
            prediction_loss_test=train_res['prediction_loss'],
            landscape_loss_test=train_res['landscape_loss'],
            acc_test=test_res['accuracy'],

            epoch_duration=time_ep
        ))
        values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['loss'],
                  test_res['accuracy'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

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

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--landscape_dimensions', type=int, default=2,
                        help='The number of dimensions in the landscape we\'re searching for. '
                             'E.g. 2 for a "normal" image')
    parser.set_defaults(orthonormal_base=False)
    parser.add_argument('--orthonormal_base', action='store_true', dest='orthonormal_base',
                        help='')
    parser.set_defaults(learn_scaling_factor=True)
    parser.add_argument('--no_scaling_factor', action='store_false', dest='learn_scaling_factor',
                        help='Whether to learn an additional scaling factor for the image like the loss landscape '
                             'sightseeing paper does for their orthonormal base.')
    # parser.add_argument('--diversity_function', type=str, default="SquaredProbabilityDistance",
    #                     help='The difference function between networks to use as diversity measure')

    parser.add_argument('--landscape_function', type=str, default="PixelDifference2D",
                        help='The function to optimize the diversity landscape for')
    parser.add_argument('--target_image', type=str, default=None,
                        help='The name of the target image in the icons folder')
    parser.add_argument('--accuracy_weight', type=float, default=0,
                        help='The weight of the actual prediction accuracy. By default, this is 0, so the network only'
                             'tries to attain the diversity image without improving the actual prediction accuracy')

    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')

    args = parser.parse_args()
    args = wandb_utils.init_wandb(args)
    main(args)
