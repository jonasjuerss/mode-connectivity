import numpy as np
import os
import torch
import torch.nn.functional as F

import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking =True)
        target = target.cuda(non_blocking =True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(non_blocking =True)
        target = target.cuda(non_blocking =True)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def test_extensive(test_loader, model, criterion, regularizer=None, t_refs = [0, 1], **kwargs):
    """
    computes additional metrics in the test loop including disagreement rate, the added true rate, and the added true rate compared to an optimal ensemble of the reference models (predictions none of the reference models made true)

    Paramters:
        test_loader : loads test data
        model : model to be evaluated
        criterion : function to return NLL (cross entropy or nll) depending on whether output is logits or probs
        regularizer : function that takes model and returns penalty term
        ref_models : List containing models the model is to be compared to #predictions are being made again and again for them - might optimize to store
    
    Returns:
        Dict containing the test metrics
    """

    

    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    disagreements = np.array([0.0] * len(t_refs)) #holds amount of samples the model disagrees with for each reference model
    added_trues = np.array([0.0] * len(t_refs)) #holds amount of true predictions that the reference model failed on
    added_trues_to_all = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(non_blocking =True)
        target = target.cuda(non_blocking =True)

        output = model(input, **kwargs)
        ref_outputs = [model(input, t_ref) for t_ref in t_refs]

        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct_pred = pred.eq(target.data.view_as(pred))
        correct += correct_pred.sum().item()

        ref_preds = [ref_output.data.argmax(1, keepdim=True) for ref_output in ref_outputs]
        ref_disagreements = np.array([sum(pred != ref_pred).item() for ref_pred in ref_preds])
        disagreements += ref_disagreements

        ref_corrects = [ref_pred.eq(target.data.view_as(ref_pred)) for ref_pred in ref_preds]
        added_trues += np.array([sum(torch.logical_and(ref_correct, ~correct_pred)).item() for ref_correct in ref_corrects])

        ref_corrects = np.array([x.cpu().detach().numpy() for x in ref_corrects])
        any_ref_correct = np.apply_along_axis(np.any, 0, ref_corrects)
        correct_pred = correct_pred.cpu().detach().numpy()
        added_trues_to_all += sum(correct_pred & ~ any_ref_correct)


        N = len(test_loader.dataset)

    return {
        'nll': nll_sum / N,
        'loss': loss_sum / N,
        'accuracy': correct * 100.0 / N,
        "disagreement" : disagreements / N,
        "new_true" : added_trues / N,
        "new_true_total" : added_trues_to_all / N
    }


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(non_blocking =True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
