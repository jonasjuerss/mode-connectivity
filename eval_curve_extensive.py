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
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if(device == "cpu"):
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.ckpt)


    model.load_state_dict(checkpoint['model_state'])

    criterion = F.cross_entropy
    regularizer = curves.l2_regularizer(args.wd)

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_nll = np.zeros(T)
    tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll = np.zeros(T)
    te_acc = np.zeros(T)
    tr_err = np.zeros(T)
    te_err = np.zeros(T)
    dl = np.zeros(T) #euclidean distance from the previous point in the weight space (includes biases)

    #measures based on test set
    disagr_start = np.zeros(T)
    disagr_end = np.zeros(T)
    added_trues_start = np.zeros(T)
    added_trues_end = np.zeros(T)
    added_trues_all = np.zeros(T)
    
    previous_weights = None

    columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)',"Disagreement t=0", "Disagreement t=1", "Added trues t=0","Added trues t=1","Added trues all"]

    evaluation_details = []

    t = torch.tensor([0.0]).to(device)
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()

        utils.update_bn(loaders['train'], model, t=t)
        tr_res = utils.test_extensive(loaders['train'], model, criterion, regularizer, t=t)
        te_res = utils.test_extensive(loaders['test'], model, criterion, regularizer, t=t)
        tr_loss[i] = tr_res['loss']
        tr_nll[i] = tr_res['nll']
        tr_acc[i] = tr_res['accuracy']
        tr_err[i] = 100.0 - tr_acc[i]
        te_loss[i] = te_res['loss']
        te_nll[i] = te_res['nll']
        te_acc[i] = te_res['accuracy']
        te_err[i] = 100.0 - te_acc[i]
        disagr_start[i] = te_res["disagreement_rates"][0]
        disagr_end[i] = te_res["disagreement_rates"][1]
        added_trues_start[i] = te_res["new_trues"][0]
        added_trues_end[i] = te_res["new_trues"][1]
        added_trues_all[i] = te_res["new_true_total"]

        values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i], disagr_start[i], disagr_end[i], added_trues_start[i], added_trues_end[i], added_trues_all[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        evaluation_details.append(values)

    for i in range(len(evaluation_details)):
        for j in range(len(evaluation_details[i])):
            if(type(evaluation_details[i][j]) == torch.Tensor):
                evaluation_details[i][j].detach().cpu().numpy()
    evaluation_details = np.array(evaluation_details)
    
    def stats(values, dl):
        min = np.min(values)
        max = np.max(values)
        avg = np.mean(values)
        int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
        return min, max, avg, int

    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

    disagr_start_min, disagr_start_max, disagr_start_avg, disagr_start_int = stats(disagr_start, dl)
    disagr_end_min, disagr_end_max, disagr_end_avg, disagr_end_int = stats(disagr_end, dl)
    added_trues_start_min, added_trues_start_max, added_trues_start_avg, added_trues_start_int = stats(added_trues_start, dl)
    added_trues_end_min, added_trues_end_max, added_trues_end_avg, added_trues_end_int = stats(added_trues_end, dl)
    added_trues_all_min, added_trues_all_max, added_trues_all_avg, added_trues_all_int = stats(added_trues_all, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
        ['disagreement with start (%)', disagr_start[0], disagr_start[-1], disagr_start_min, disagr_start_max, disagr_start_avg, disagr_start_int],
        ['disagreement with end (%)', disagr_end[0], disagr_end[-1], disagr_end_min, disagr_end_max, disagr_end_avg, disagr_end_int],
        ['added trues start (%)', added_trues_start[0], added_trues_start[-1], added_trues_start_min, added_trues_start_max, added_trues_start_avg, added_trues_start_int],
        ['added trues end (%)', added_trues_end[0], added_trues_end[-1], added_trues_end_min, added_trues_end_max, added_trues_end_avg, added_trues_end_int],
        ['added trues over all (%)', added_trues_all[0], added_trues_all[-1], added_trues_all_min, added_trues_all_max, added_trues_all_avg, added_trues_all_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

    np.savez(os.path.join(args.dir, 'curve_eval_details.npz', evaluation_details = evaluation_details))
    np.savez(
        os.path.join(args.dir, 'curve.npz'),
        ts=ts,
        dl=dl,
        tr_loss=tr_loss,
        tr_loss_min=tr_loss_min,
        tr_loss_max=tr_loss_max,
        tr_loss_avg=tr_loss_avg,
        tr_loss_int=tr_loss_int,
        tr_nll=tr_nll,
        tr_nll_min=tr_nll_min,
        tr_nll_max=tr_nll_max,
        tr_nll_avg=tr_nll_avg,
        tr_nll_int=tr_nll_int,
        tr_acc=tr_acc,
        tr_err=tr_err,
        tr_err_min=tr_err_min,
        tr_err_max=tr_err_max,
        tr_err_avg=tr_err_avg,
        tr_err_int=tr_err_int,
        te_loss=te_loss,
        te_loss_min=te_loss_min,
        te_loss_max=te_loss_max,
        te_loss_avg=te_loss_avg,
        te_loss_int=te_loss_int,
        te_nll=te_nll,
        te_nll_min=te_nll_min,
        te_nll_max=te_nll_max,
        te_nll_avg=te_nll_avg,
        te_nll_int=te_nll_int,
        te_acc=te_acc,
        te_err=te_err,
        te_err_min=te_err_min,
        te_err_max=te_err_max,
        te_err_avg=te_err_avg,
        te_err_int=te_err_int,

        disagr_start=disagr_start, 
        disagr_start_min=disagr_start_min, 
        disagr_start_max=disagr_start_max, 
        disagr_start_avg=disagr_start_avg, 
        disagr_start_int=disagr_start_int,
        disagr_end=disagr_end,
        disagr_end_min=disagr_end_min, 
        disagr_end_max=disagr_end_max, 
        disagr_end_avg=disagr_end_avg,
        disagr_end_int=disagr_end_int,
        added_trues_start=added_trues_start, 
        added_trues_start_min=added_trues_start_max,
        added_trues_start_max=added_trues_start_max, 
        added_trues_start_avg= added_trues_start_avg,
        added_trues_start_int=added_trues_start_int,
        added_trues_end=added_trues_end, 
        added_trues_end_min=added_trues_end_min, 
        added_trues_end_max=added_trues_end_max, 
        added_trues_end_avg=added_trues_end_avg, 
        added_trues_end_int=added_trues_end_int,
        added_trues_all=added_trues_all, 
        added_trues_all_min=added_trues_all_min, 
        added_trues_all_max=added_trues_all_max, 
        added_trues_all_avg=added_trues_all_avg, 
        added_trues_all_int=added_trues_all_int
    )

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
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size (default: 64)')
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


