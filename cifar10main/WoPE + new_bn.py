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
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from opacus.utils import module_modification

from gradient_utils import get_first_batch_data, prune_grad_percentage, copy_model, prune_grad_val, trunc_grad, normalize_grad
from gradient_utils import model_mix, model_momemtum_mix, dot_product, recompute_bn_gradient, generate_mask, multiply_mask
from mix_data_utils import mixup_data, mixup_public_data, mixup_criterion

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

# print("resnet20, method 0 L2, gap =0.15, randommix, grouping=3, clip_norm, 1 batch_size: 600, noise 0.004, start_rate 0.01")
# print("resnet20, comparison 0.0015 L_1 3 0.01 tau = 0 ")


dev = 0
device = torch.device('cuda:0')

batch_select = 4
true_batch = 5000 / batch_select
start_lr = 0.2
gap_rate = 0.1
num_epoch = 120
noise_scale = 0.005

use_SVHN = False
use_CIFAR10 = True
use_FMNIST = False

keep_bn = True  # True: use batchnorm layers; False: use groupnorm layers
use_mix = True  # use random mix
use_norm = True  # normalize gradient
clip_type = 2.0
clip_norm = 6.25

mix_half = False
compute_varmean = False
use_merge = False
merge_epoch = 20
use_prune_after_norm = False
prune_percentage_2 = 0
use_public_mask = False
mask_percentage = 90

use_prune = False  # prune per sample gradient before adding them
prune_percentage = 0

use_trunc = False  # truncate per sample gradient before adding them
max_val = 1.2

use_expand = False  # generate multiple data samples from few samples
expand_batch_size = 1464
use_public_expand = False  # combine few samples into multiple samples using public samples
public_batch_size = 20
use_precompute_bn = False  # replace bn gradient with precompute_bn
calc_bn_size = 50

train_fc_only = False  # only train fully connected layer

print(torch.__version__, torch.version.cuda)
print("Learning Rate", start_lr)
print("Pruning:", prune_percentage, prune_percentage_2, "Use Mix", use_mix, "Gap", gap_rate, "CLIP NORM", clip_norm, "Grouping",
      batch_select, "True Batch", true_batch, "Noise", noise_scale, "Num of Epochs", num_epoch, "Device", dev)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=num_epoch, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=batch_select, type=int,
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
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1, dev
    global noise_scale, clip_norm, use_public_expand
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_r50 = False
    if use_r50:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        model2 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        store_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    else:
        model = resnet.__dict__[args.arch]()
        model2 = resnet.__dict__[args.arch]()
        store_model = resnet.__dict__[args.arch]()

    # optionally resume from a checkpoint
    if keep_bn:
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                temp_dict = model.state_dict()
                for name in temp_dict.keys():
                    if (temp_dict[name].shape != checkpoint['state_dict'][name].shape):
                        print("alert: stored model differs declared model in layer ", name)
                        checkpoint['state_dict'][name] = temp_dict[name]
                model.load_state_dict(checkpoint['state_dict'])
                model2.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model = module_modification.convert_batchnorm_modules(model)
        model2 = module_modification.convert_batchnorm_modules(model2)

    model.cuda(device)
    model2.cuda(device)
    store_model.cuda(device)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if use_SVHN:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root='./data', split='train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root='./data', split='test', transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=1000, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif use_CIFAR10:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_select, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=2000, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif use_FMNIST:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_select, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=2000, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    public_input, public_target = get_first_batch_data(val_loader)
    public_input, public_target = public_input.cuda(device), public_target.cuda(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)
    # criterion = nn.MSELoss().cuda(device)

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer2 = torch.optim.SGD(model2.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], last_epoch=args.start_epoch - 1)
    #lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[100], last_epoch=args.start_epoch - 1)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992, last_epoch=args.start_epoch - 1)
    lr_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992, last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.5

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    norm_vec, var_vec, acc_vec = [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        if use_merge and (epoch + 1) % merge_epoch == 0:
            copy_model(model2, model)
        '''
        global gap_rate
        if epoch < 30:
           gap_rate= 0.1
        elif epoch  < 60:
           gap_rate = 0.075
        else:
           gap_rate = 0.05
        '''

        # clip_norm = 4
        # else:
        # noise_scale = 0
        # clip_norm = 8

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, model2, store_model, criterion, optimizer, optimizer2, public_input, public_target,
              epoch, norm_vec, var_vec)
        if compute_varmean:
            print("norm vector: ", norm_vec)
            print("variance vector: ", var_vec)

        lr_scheduler.step()
        lr_scheduler2.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        acc_vec.append(prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'modelsvhn.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpointsvhn.th'))
    if compute_varmean:
        print("final mean: ", norm_vec)
        print("final variance: ", var_vec)
    print("accuracy list: ", acc_vec)


def train(train_loader, model, model2, store_model, criterion,
          optimizer, optimizer2, public_input, public_target, epoch, norm_vec, var_vec):
    """
        Run one train epoch
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model2.train()
    store_model.train()

    end = time.time()
    lr_temp = optimizer.param_groups[0]['lr']
    tau = lr_temp * gap_rate
    # print(tau)
    store_grad = {}
    true_grad = {}
    mask = {}
    input_group = torch.empty(0).cuda(device)
    target_group = torch.empty(0).cuda(device)

    grad_list = {}
    sample_number = 0
    for i, (input, target) in enumerate(train_loader):
        '''
        if i % true_batch == 0 and epoch % 2 == 0:
            batch_size = public_input.size()[0]
            index = torch.randint(1, 1000, (public_batch_size,)).cuda(device)
            index0 = torch.randint(1, 1000, (public_batch_size,)).cuda(device)
        elif i % true_batch == 0 and epoch % 2 == 1:
            batch_size = public_input.size()[0]
            index = torch.randint(1000, 2000, (public_batch_size,)).cuda(device)
            index0 = torch.randint(1000, 2000, (public_batch_size,)).cuda(device)
        '''
        index = torch.randint(1, 1000, (public_batch_size,)).cuda(device)
        index0 = torch.randint(1, 1000, (public_batch_size,)).cuda(device)
        temp_model = model
        temp_model2 = model2
        temp_optimizer = optimizer

        if i % true_batch == 0:
            if use_public_mask:
                mask = generate_mask(temp_model, temp_optimizer, criterion,
                    public_input, public_target, mask_percentage)
            else:
                mask = {}

        '''
        if math.floor(i / true_batch) % 2 == 0:
            temp_model = model
            temp_model2 = model2
            temp_optimizer = optimizer
        else:
            temp_model = model2
            temp_model2 = model
            temp_optimizer = optimizer2
        '''
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(device)
        input_var = input.cuda(device)
        target_var = target
        input_group = torch.cat((input_group, input_var), 0)
        target_group = torch.cat((target_group, target_var), 0)

        if mix_half:
            temp_dict = copy.deepcopy(temp_model.state_dict())
            temp_dict2 = copy.deepcopy(temp_model2.state_dict())
            for name in temp_dict:
                temp_dict2[name] = (temp_dict[name] + temp_dict2[name]) / 2
            temp_model.load_state_dict(temp_dict2)

        # calculate the gradient
        if use_expand:
            times = int(expand_batch_size / batch_select)
            for j in range(times):
                inputs, targets_a, targets_b, targets_c, lam = mixup_data(input_var, target_var)
                outputs = temp_model(inputs)
                if j == 0:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, targets_c, lam) / times
                else:
                    loss += mixup_criterion(criterion, outputs, targets_a, targets_b, targets_c, lam) / times
        elif use_public_expand:
            inputs, y1, y2, y3, lam = mixup_public_data(input_var, target_var, public_input, public_target, index,
                                                        index0, public_batch_size, batch_select)
            outputs = temp_model(inputs)
            loss = mixup_criterion(criterion, outputs, y1, y2, y3, lam)
        else:
            outputs = temp_model(input_var)
            loss = criterion(outputs, target_var)

        temp_optimizer.zero_grad()
        loss.backward()

        if compute_varmean and sample_number < 200:
            for name, param in temp_model.named_parameters():
                if name not in grad_list:
                    grad_list[name] = param.grad.unsqueeze(0)
                else:
                    grad_list[name] = torch.cat((grad_list[name], param.grad.unsqueeze(0)), 0)
            sample_number += 1
        if (i + 1) % true_batch == 0:
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
                grad_list = {}
                sample_number = 0

        # modify the gradient
        if use_public_mask:
            multiply_mask(temp_model, mask)
        if use_prune:
            prune_grad_percentage(temp_model, prune_percentage)
        if use_trunc:
            trunc_grad(temp_model, max_val)
        if use_norm:
            normalize_grad(temp_model, clip_norm)

        # sum the gradient with other samples and put it in store_Grad
        for name, param in temp_model.named_parameters():
            if train_fc_only and "linear" not in name:
                continue
            if name not in store_grad:
                store_grad[name] = param.grad / true_batch
            else:
                store_grad[name] += param.grad / true_batch

        if mix_half:
            temp_model.load_state_dict(temp_dict)

        # update by calling step every true_batch rounds
        if (i + 1) % true_batch == 0:
            for name, param in temp_model.named_parameters():
                if train_fc_only and "linear" not in name:
                    continue
                if clip_type == 2.0:
                    noise = torch.randn(param.grad.shape)
                if clip_type == 1.0:
                    laplace_dist = torch.distributions.laplace.Laplace(0, 1)
                    noise = laplace_dist.sample(param.grad.shape)
                param.grad = store_grad[name] + torch.mul(noise, noise_scale).cuda(device)

            if keep_bn and use_precompute_bn:
                calc_bn_num = int(true_batch * batch_select / calc_bn_size)
                recompute_bn_gradient(temp_model, store_model, criterion, input_group, target_group,
                                      calc_bn_size, calc_bn_num,
                                      use_prune, use_trunc, use_norm, prune_percentage, max_val, clip_norm)

            if use_prune_after_norm:
                prune_grad_val(temp_model, noise_scale)
                # prune_grad_percentage(temp_model, prune_percentage_2)
            '''
            if i > 0 and (i + 1) % (args.print_freq * true_batch / 5) == 0:
                print(dot_product(temp_model, true_grad))
            '''

            if train_fc_only:
                for name, param in temp_model.named_parameters():
                    if "linear" not in name:
                        param.grad = None

            if use_mix:
                # model_mix(temp_model, temp_model2, tau)
                item_net_dict = copy.deepcopy(temp_model.state_dict())
                tau = lr_temp * gap_rate
                model_mix(temp_model, temp_model2, tau, mask)
                #model_momemtum_mix(temp_model, temp_model2, tau, mask)
                temp_model2.load_state_dict(item_net_dict)

            temp_optimizer.step()

            store_grad = {}
            true_grad = {}
            input_group = torch.empty(0).cuda(device)
            target_group = torch.empty(0).cuda(device)

        # measure accuracy and record loss
        outputs = temp_model(input_var)
        outputs = outputs.float()
        loss = loss.float()
        prec1 = accuracy(outputs.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i > 0 and (i + 1) % (args.print_freq * true_batch / 5) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Prec@1 {top1.avg:.3f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

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
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
