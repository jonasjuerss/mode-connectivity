import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils


def main(args):
    os.makedirs(args.dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    loaders, num_classes = data.loaders(
        dataset = args.dataset,
        path = args.data_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        transform_name = args.transform,
        use_test = args.use_test,
        shuffle_train=False
    )

    architecture = getattr(models, args.model)
    curve = getattr(curves, args.curve)
    model = curves.CurveSystemNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    model.cuda()
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'])

    criterion = F.cross_entropy

    cross_road = architecture.base(num_classes)
    model.export_base_parameters(cross_road, 0)
    cross_road.cuda()


    regularizer = None
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=0.05,
        weight_decay=5e-4,
        momentum=None
    )

    global_lr = 0.05
    num_epochs = 100
    
    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr
    
    for epoch in range(num_epochs):
        lr = learning_rate_schedule(global_lr, epoch, num_epochs)
        utils.adjust_learning_rate(optimizer, lr)

        utils.train(loaders['train'], model, optimizer, criterion, regularizer)

    tr_res = utils.test(loaders['train'], cross_road, criterion)
    te_res = utils.test(loaders['test'], cross_road, criterion)
    result = [[tr_res["nll"], tr_res["loss"], tr_res["accuracy"] ], [te_res["nll"], te_res["loss"], te_res["accuracy"] ]]
    result = np.array(result)
    np.savez(os.path.join(args.dir, 'experimental_fix.npz'), result = result)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN curve evaluation')
    parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                        help='training directory (default: /tmp/eval)')

    parser.add_argument('--num_points', type=int, default=61, metavar='N',
                        help='number of points on the curve (default: 61)')

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
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')

    parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                        help='checkpoint to eval (default: None)')

    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    args = parser.parse_args()
    main(args)


