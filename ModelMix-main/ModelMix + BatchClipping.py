import argparse
import os
import shutil
import time
import random
import resnet
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from opacus.utils import module_modification
from original_cifar import CIFAR10

from loader import get_model, get_data_SVHN, get_data_FMINIST, get_data_CIFAR10, get_optimizer, iid_sample
from gradient_utils import get_first_batch_data, prune_grad_percentage, copy_model, prune_grad_val, trunc_grad, normalize_grad, calc_grad_norm
from gradient_utils import model_mix, model_momemtum_mix, dot_product, recompute_bn_gradient, generate_mask, multiply_mask

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

# print("resnet20, method 0 L2, gap =0.15, randommix, grouping=3, clip_norm, 1 batch_size: 600, noise 0.004, start_rate 0.01")
# print("resnet20, comparison 0.0015 L_1 3 0.01 tau = 0 ")


dev = 0
device = torch.device('cuda:0')

# learning parameters
batch_size = 1250
num_iter = 8000
start_lr = 0.15
gap_rate = 0.1

use_SVHN = False
use_CIFAR10 = True
use_FMNIST = False

# privacy parameters
group_size = 10
group_num = batch_size / group_size
noise_scale =  0    #0.0092

keep_bn = True  # True: use batchnorm layers; False: use groupnorm layers
use_mix = True  # use random mix
use_norm = True  # normalize gradient
use_trunc = Fa  # truncate per sample gradient after clipping
clip_type = 2.0
clip_norm = 5
trunc_ratio = 0.1

# other / print parameters
compute_varmean = False
use_prune = False
prune_percentage = 0
use_merge = False
merge_iter = 1000
use_prune_after_norm = False
prune_percentage_2 = 0
train_fc_only = False  # only train fully connected layer

print(torch.__version__, torch.version.cuda)
print("Learning Rate", start_lr)
print("Use Mix", use_mix, "Gap", gap_rate,
    "Use_Norm", use_norm, "GroupSize", group_size, "BatchSize", batch_size, "CLIP NORM", clip_norm,
    "Noise", noise_scale, "Iterations", num_iter, "Device", dev)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=start_lr, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

#parser.add_argument('--resume', default='save_temp/cifar10-2500-24.th', type=str, metavar='PATH', help ='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=500)
best_prec1 = 0


def main():
    global args, best_prec1, dev
    global noise_scale, clip_norm
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_r50 = False
    model, model2 = get_model(args.arch, args.resume, keep_bn, use_r50)
    model = model.cuda(device)
    model2 = model2.cuda(device)

    if use_SVHN:
        train_set, test_set = get_data_SVHN()
    elif use_FMNIST:
        train_set, test_set = get_data_FMINIST()
    elif use_CIFAR10:
        train_set, test_set = get_data_CIFAR10()

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=2000, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda(device)
    # criterion = nn.MSELoss().cuda(device)

    if args.half:
        model.half()
        criterion.half()

    optimizer, lr_scheduler = get_optimizer(model=model, lr=args.lr, momentum=args.momentum,
        decay=args.weight_decay, gamma=0.993)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    norm_vec, var_vec, acc_vec = [], [], []
    change_iter = int(len(train_set) / batch_size)
    train_prec = 0
    for iters in range(1, num_iter):
        if use_merge and (iters + 1) % merge_iter == 0:
            copy_model(model2, model)

        all_inputs, all_targets = iid_sample(train_set, batch_size / len(train_set))
        train_prec += train(all_inputs, all_targets, model, model2, criterion, optimizer, iters, norm_vec, var_vec)
        if iters % (int(change_iter / 5)) == 0:
            print("iteration ", iters, " train accuracy: ", train_prec.item() / int(change_iter / 5))
            train_prec = 0
        if compute_varmean:
            print("norm vector: ", norm_vec[-1], "variance vector: ", var_vec[-1])
        if iters % change_iter == 0:
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            prec1 = validate(val_loader, model, criterion)
            acc_vec.append(prec1)
            lr_scheduler.step()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if iters > 0 and iters % args.save_every == 0:
                save_checkpoint({
                    'epoch': iters / change_iter,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'modelsvhn.th'))

    if compute_varmean:
        print("final mean: ", norm_vec)
        print("final variance: ", var_vec)
    print("accuracy list: ", acc_vec)


def train(all_inputs, all_targets, model, model2, criterion,
          optimizer, iters, norm_vec, var_vec):
    """
        Run one train epoch
    """

    # switch to train mode
    model.train()
    model2.train()

    lr_temp = optimizer.param_groups[0]['lr']
    tau = lr_temp * gap_rate
    store_grad = {}
    all_inputs, all_targets = all_inputs.cuda(device), all_targets.cuda(device)

    grad_list = {}
    sample_number = 0
    iter_num = all_inputs.shape[0] / group_size
    for i in range(int(iter_num)):
        if (i + 1) * group_size > all_inputs.shape[0]:
            break
        inputs = all_inputs[i * group_size: (i + 1) * group_size]
        targets = all_targets[i * group_size: (i + 1) * group_size].type(torch.LongTensor).cuda(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        # we don't have enough storage to store all per-sample gradients, instead only store 200 of those
        if compute_varmean and sample_number < 200:
            for name, param in model.named_parameters():
                if name not in grad_list:
                    grad_list[name] = param.grad.unsqueeze(0)
                else:
                    grad_list[name] = torch.cat((grad_list[name], param.grad.unsqueeze(0)), 0)
            sample_number += 1

        if use_prune:
            prune_grad_percentage(model, prune_percentage)
        if use_trunc:
            norm_val = calc_grad_norm(model)
            trunc_grad(model, norm_val * trunc_ratio)
        if use_norm:
            normalize_grad(model, clip_norm)

        # sum the gradient with other samples and put it in store_Grad
        for name, param in model.named_parameters():
            if train_fc_only and "linear" not in name:
                continue
            if name not in store_grad:
                store_grad[name] = param.grad / group_num
            else:
                store_grad[name] += param.grad / group_num

    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        if clip_type == 2.0:
            noise = torch.randn(param.grad.shape)
        if clip_type == 1.0:
            laplace_dist = torch.distributions.laplace.Laplace(0, 1)
            noise = laplace_dist.sample(param.grad.shape)
        param.grad = store_grad[name] + torch.mul(noise, noise_scale).cuda(device)

    if use_prune_after_norm:
        prune_grad_val(model, noise_scale)
        # prune_grad_percentage(model, prune_percentage_2)

    if train_fc_only:
        for name, param in model.named_parameters():
            if "linear" not in name:
                param.grad = None

    if use_mix:
        item_net_dict = copy.deepcopy(model.state_dict())
        tau = lr_temp * gap_rate
        model_mix(model, model2, tau)
        model2.load_state_dict(item_net_dict)

    optimizer.step()

    if compute_varmean:
        mean_norm = torch.tensor(0.0).cuda(device)
        var_norm = torch.tensor(0.0).cuda(device)
        for name in grad_list:
            mean_grad = torch.sum(grad_list[name], 0).unsqueeze(0) / sample_number
            mean_norm += mean_grad.norm(2) ** 2
            var_grad = grad_list[name] - mean_grad.expand(grad_list[name].shape)
            var_norm += (var_grad.norm(2) ** 2) / sample_number
            norm_vec.append(torch.sqrt(mean_norm).item())
        var_vec.append(torch.sqrt(var_norm).item())

    # measure accuracy and record loss
    all_outputs = model(all_inputs)
    all_outputs = all_outputs.float()
    prec1 = accuracy(all_outputs.data, all_targets)[0]
    return prec1


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    momentum = {}
    for name, m in model.named_modules():
        if "bn" in name:
            m.reset_running_stats()
            m.train()
            momentum[name] = m.momentum
            m.momentum = None

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device)
            input_var = input.cuda(device)
            target_var = target.cuda(device)
            output = model(input_var)

    for name, m in model.named_modules():
        if name in momentum:
            m.momentum = momentum[name]

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device)
            input_var = input.cuda(device)
            target_var = target.cuda(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_len = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_len))
    return res


if __name__ == '__main__':
    main()