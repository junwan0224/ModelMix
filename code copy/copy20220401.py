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

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

# print("resnet20, method 0 L2, gap =0.15, randommix, grouping=3, clip_norm, 1 batch_size: 600, noise 0.004, start_rate 0.01")
# print("resnet20, comparison 0.0015 L_1 3 0.01 tau = 0 ")


dev = 1
device = torch.device('cuda:1')

keep_bn = True
use_prune = False
use_trunc = False
use_norm = True
use_expand = False
use_public_expand = False
use_prune_after_norm = False
use_mix = True
use_precompute_bn = False
seperate_clip = False
use_bn_ratio = False
use_merge = True
train_fc_only = False

batch_select = 5
public_batch_size = 30
expand_batch_size = 20
calc_bn_size = 50
true_batch = 5000/batch_select
merge_epoch = 20
prune_percentage = 0
prune_percentage_2 = 0
noise_scale = 0.0025
bn_step_factor = 1.0
max_val = 1.2
clip_method = 1
clip_type = 2.0
clip_norm = 2.4
start_lr = 0.1
gap_rate = 0.075
num_epoch = 200
print(torch.__version__, torch.version.cuda)
print("Learning Rate", start_lr, bn_step_factor)
print("Pruning:", prune_percentage, prune_percentage_2, "Gap", gap_rate, "CLIP NORM", clip_norm, "Grouping",
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
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 50)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--resume', default='save_temp/M-4-0.15-100.th', type=str, metavar='PATH', help ='path to latest checkpoint (default: none)')

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

def get_sample_data (data_loader):
    for i, (input, target) in enumerate(data_loader):
        if i >= 1:
            break
        return input.cuda(device), target.cuda(device)


def main():
    global args, best_prec1, dev
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
    '''
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
    '''

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

    public_input, public_target = get_sample_data(val_loader)


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

    optimizer3 = torch.optim.SGD(store_model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], last_epoch=args.start_epoch - 1)
    #lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[100], last_epoch=args.start_epoch - 1)
    lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[50, 80, 110],
                                                         last_epoch=args.start_epoch - 1)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.993, last_epoch=args.start_epoch - 1)

    lr_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.993, last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.5

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    global noise_scale, clip_norm

    for epoch in range(args.start_epoch, args.epochs):
        if use_merge and (epoch + 1) % merge_epoch == 0:
            copy_model(model2, model)
        # if epoch < 40:
        # noise_scale = 0.02
        # clip_norm = 4
        # else:
        # noise_scale = 0
        # clip_norm = 8

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        '''
        global use_public_expand
        if epoch < 50:
            use_public_expand = False
        else:
            use_public_expand = True
        '''
        train(use_public_expand, train_loader, model, model2, store_model, criterion, optimizer, optimizer2, optimizer3, public_input, public_target, epoch)
        lr_scheduler.step()
        lr_scheduler2.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))


def grad_modify(model, use_prune, use_trunc, use_norm):
    percent_noise_scale = 0
    if use_prune:
        all_param = torch.empty(0).cuda(device)
        sum_grad = {}
        for name, param in model.named_parameters():
            if train_fc_only and "linear" not in name:
                continue
            laplace_dist = torch.distributions.laplace.Laplace(0, 1)
            noise1 = laplace_dist.sample(param.grad.shape).cuda(device) * percent_noise_scale
            sum_grad[name] = noise1 + param.grad.abs()
            all_param = torch.cat((all_param, torch.flatten(sum_grad[name])), 0)
        percentile_value = np.percentile(all_param.cpu().numpy(), prune_percentage)

        for name, param in model.named_parameters():
            if train_fc_only and "linear" not in name:
                continue
            param.grad = torch.where(sum_grad[name] < torch.tensor(percentile_value).cuda(device),
                                     torch.tensor(0.0).cuda(device), param.grad)

    if use_trunc:
        for name, param in model.named_parameters():
            param.grad.clamp(min=-max_val, max=max_val)

    if use_norm:
        sum_norm = torch.tensor(0.0).cuda(device)
        for name, param in model.named_parameters():
            norm_val = param.grad.norm(2)
            sum_norm += norm_val * norm_val
        sum_norm = torch.sqrt(sum_norm)
        if sum_norm > torch.tensor(clip_norm):
            for name, param in model.named_parameters():
                param.grad /= sum_norm / clip_norm


def model_mix(model, model2, tau):
    temp_dict = model.state_dict()
    temp_dict2 = model2.state_dict()
    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        dictShape = temp_dict[name].shape
        gap_opr = torch.abs(temp_dict[name] - temp_dict2[name])
        if torch.min(gap_opr) < tau:
            gap_opr = torch.clamp(gap_opr, min=0, max=tau)
            gap_opr = torch.add(-gap_opr, tau)
            sign_opr = torch.sign(temp_dict[name] - temp_dict2[name])
            #sign_opr[sign_opr == 0] = 1
            #print(name, tau, torch.sum(torch.abs(sign_opr)), torch.numel(sign_opr))
            temp_dict[name] += gap_opr * sign_opr / 2
            temp_dict2[name] -= gap_opr * sign_opr / 2

        oness = torch.ones(dictShape).cuda(device)
        alpha1 = torch.rand(dictShape).cuda(device)
        temp_dict[name] = alpha1 * temp_dict[name] + (oness - alpha1) * temp_dict2[name]
    model.load_state_dict(temp_dict)

def mixup_data(x, y, alpha = 0.1):
    if alpha > 0:
        #lam = np.random.beta(alpha, alpha)
        lam = 0.1
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(device)
    index0 = torch.randperm(batch_size).cuda(device)
    mixed_x = (1 - 2*lam) * x + lam * x[index, :] + lam* x[index0, :]
    y_a, y_b, y_c = y, y[index], y[index0]
    return mixed_x, y_a, y_b, y_c, lam

def mixup_public_data(private_x, private_y, public_x, public_y, index, index0, alpha = 0.1):
    if alpha > 0:
        #lam = np.random.beta(alpha, alpha)
        lam = 0.1
    else:
        lam = 1

    batch_size = public_x.size()[0]
    repeat_num = int(public_batch_size / batch_select)
    #index = torch.randint(batch_size, (public_batch_size,)).cuda(device)
    x2, y2 = public_x[index], public_y[index]
    #index0 = torch.randint(batch_size, (public_batch_size,)).cuda(device)
    x3, y3 = public_x[index0], public_y[index0]
    x1, y1 = private_x.repeat(repeat_num, 1, 1, 1), private_y.repeat(repeat_num)
    #print(x1.size(), x2.size(), x3.size())
    mixed_x = (1 - 2*lam) * x1 + lam * x2 + lam * x3
    return mixed_x, y1, y2, y3, lam

def mixup_criterion(criterion, pred, y_a, y_b, y_c, lam):
    return (1 - 2*lam) * criterion(pred, y_a) + lam * criterion(pred, y_b) + lam*criterion(pred, y_c)

def copy_model(model, model2):
    model.load_state_dict(copy.deepcopy(model2.state_dict()))


def train(use_public_expand, train_loader, model, model2, store_model, criterion,
    optimizer, optimizer2, optimizer3, public_input, public_target, epoch):
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
    input_group = torch.empty(0).cuda(device)
    target_group = torch.empty(0).cuda(device)

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

        if use_expand:
            times = int(expand_batch_size / batch_select)
            for j in range(times):
                inputs, targets_a, targets_b, targets_c, lam = mixup_data(input_var, target_var)
                outputs = temp_model(inputs)
                if j == 0:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, targets_c,lam) / times
                else:
                    loss += mixup_criterion(criterion, outputs, targets_a, targets_b, targets_c,lam) / times
        elif use_public_expand:
            inputs, y1, y2, y3,lam = mixup_public_data(input_var, target_var, public_input, public_target, index, index0)
            outputs = temp_model(inputs)
            loss = mixup_criterion(criterion, outputs, y1, y2, y3, lam)
        else:
            outputs = temp_model(input_var)
            loss = criterion(outputs, target_var)

        temp_optimizer.zero_grad()
        loss.backward()
        # temp_optimizer.step()

        grad_modify(temp_model, use_prune, use_trunc, use_norm)

        for name, param in temp_model.named_parameters():
            if train_fc_only and "linear" not in name:
                continue
            if name not in store_grad:
                store_grad[name] = param.grad / true_batch
            else:
                store_grad[name] += param.grad / true_batch

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
                bn_store_norm = {}
                copy_model(store_model, temp_model)
                for j in range(calc_bn_num):
                    new_input = input_group[j * calc_bn_size: (j + 1) * calc_bn_size]
                    new_target = target_group[j * calc_bn_size: (j + 1) * calc_bn_size].long()
                    store_output = store_model(new_input)
                    store_loss = criterion(store_output, new_target)
                    store_model.zero_grad()
                    store_loss.backward()
                    for name, param in store_model.named_parameters():
                        if name not in true_grad:
                            true_grad[name] = param.grad / calc_bn_num
                        else:
                            true_grad[name] += param.grad / calc_bn_num
                    grad_modify(store_model, use_prune, use_trunc, use_norm)

                    for name, param in store_model.named_parameters():
                        if "bn" not in name:
                            continue
                        if name not in bn_store_norm:
                            bn_store_norm[name] = param.grad / calc_bn_num
                        else:
                            bn_store_norm[name] += param.grad / calc_bn_num

                for name, param in temp_model.named_parameters():
                    if "bn" in name:
                        param.grad = bn_store_norm[name]

            if use_prune_after_norm and not train_fc_only:
                all_param = torch.empty(0).cuda(device)
                for name, param in temp_model.named_parameters():
                    # if "bn" in name or "linear" in name:
                    #    continue
                    all_param = torch.cat((all_param, torch.flatten(param.grad.abs())), 0)
                percentile_value = np.percentile(all_param.cpu().numpy(), prune_percentage_2)
                percentile_value = 0.5 * noise_scale
                # print("scale is: ", percentile_value, noise_scale)

                for name, param in temp_model.named_parameters():
                    # if "bn" in name or "linear" in name:
                    #    continue
                    param.grad = torch.where(param.grad.abs() < torch.tensor(percentile_value).cuda(device),
                                             torch.tensor(0.0).cuda(device), param.grad)
                '''
                if i > 0 and (i + 1) % (args.print_freq * true_batch / 5) == 0:
                    product = torch.tensor(0.0).cuda(device)
                    true_norm = torch.tensor(0.0).cuda(device)
                    store_norm = torch.tensor(0.0).cuda(device)
                    for name, param in temp_model.named_parameters():
                        product += torch.sum(param.grad * true_grad[name])
                        norms = true_grad[name].norm(p=2)
                        true_norm += norms * norms
                        norms = param.grad.norm(p=2)
                        store_norm += norms * norms
                    print(product / torch.sqrt(true_norm * store_norm))
                '''

            if train_fc_only:
                for name, param in temp_model.named_parameters():
                    if "linear" not in name:
                        param.grad = None

            if use_mix:
                #model_mix(temp_model, temp_model2, tau)
                item_net_dict = copy.deepcopy(temp_model.state_dict())
                tau = lr_temp * gap_rate
                model_mix(temp_model, temp_model2, tau)


            temp_optimizer.step()

            if use_mix:
                temp_model2.load_state_dict(item_net_dict)

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