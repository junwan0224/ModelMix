import torch
import numpy as np
import math
import copy
import random
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_first_batch_data(data_loader):
    for i, (input, target) in enumerate(data_loader):
        if i >= 1:
            break
        return input, target


def prune_grad_percentage(model, prune_percentage, prune_noise_scale=0, train_fc_only=False):
    dev = next(model.parameters()).device
    all_param = torch.empty(0).cuda(dev)
    sum_grad = {}
    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        if prune_noise_scale != 0:
            laplace_dist = torch.distributions.laplace.Laplace(0, 1)
            noise1 = laplace_dist.sample(param.grad.shape).cuda(dev) * prune_noise_scale
            sum_grad[name] = noise1 + param.grad.abs()
        else:
            sum_grad[name] = param.grad.abs()
        all_param = torch.cat((all_param, torch.flatten(sum_grad[name])), 0)
    percentile_value = np.percentile(all_param.cpu().numpy(), prune_percentage)

    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        param.grad = torch.where(sum_grad[name] < torch.tensor(percentile_value).cuda(dev),
                                 torch.tensor(0.0).cuda(dev), param.grad)


def prune_grad_val(model, prune_val, train_fc_only=False):
    dev = next(model.parameters()).device
    for name, param in model.named_parameters():
        param.grad = torch.where(param.grad.abs() < torch.tensor(prune_val).cuda(dev),
                                 torch.tensor(0.0).cuda(dev), param.grad)


def trunc_grad(model, max_val, train_fc_only=False):
    dev = next(model.parameters()).device
    clamp_count, total = torch.tensor(0).cuda(dev), torch.tensor(0).cuda(dev)
    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        clamp_count += torch.count_nonzero(param.grad.abs() > max_val)
        '''
        total += torch.numel(param.grad)
        print(name)
        all_param = torch.flatten(param.grad.abs()).cpu().numpy()
        print_percentage = [100]
        for p in print_percentage:
            percentile_value = np.percentile(all_param, p)
            print("percent ", p, ": ", percentile_value / 5)
        '''
        param.grad = param.grad.clamp(min=-max_val, max=max_val)
    #print(clamp_count / total)

def calc_grad_norm(model):
    dev = next(model.parameters()).device
    sum_norm = torch.tensor(0.0).cuda(dev)
    for name, param in model.named_parameters():
        norm_val = param.grad.norm(2)
        sum_norm += norm_val * norm_val
    sum_norm = torch.sqrt(sum_norm)
    return sum_norm


def normalize_grad(model, clip_norm):
    dev = next(model.parameters()).device
    sum_norm = torch.tensor(0.0).cuda(dev)
    for name, param in model.named_parameters():
        norm_val = param.grad.norm(2)
        sum_norm += norm_val * norm_val
    sum_norm = torch.sqrt(sum_norm)
    if sum_norm > torch.tensor(clip_norm):
        for name, param in model.named_parameters():
            param.grad /= sum_norm / clip_norm


def generate_mask(model, optimizer, criterion, inputs, targets, mask_percentage):
    dev = next(model.parameters()).device
    all_param = torch.empty(0).cuda(dev)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    mask = {}
    for name, param in model.named_parameters():
        all_param = torch.cat((all_param, torch.flatten(param.grad.abs())), 0)
    percentile_value = np.percentile(all_param.cpu().numpy(), mask_percentage)
    for name, param in model.named_parameters():
        if "bn" in name or "linear" in name or "fc" in name: continue # uncomment this line
        mask[name] = torch.where(param.grad.abs() < torch.tensor(percentile_value).cuda(dev),
                                 torch.tensor(0.0).cuda(dev), torch.tensor(1.0).cuda(dev))
    return mask


def multiply_mask(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.grad = param.grad * mask[name]


def model_mix(model, model2, tau, mask={}, train_fc_only=False):
    dev = next(model.parameters()).device
    dev2 = next(model2.parameters()).device
    if dev != dev2:
        print("model1 and model2 are not on the same cuda device!")
        assert (False)
    temp_dict = copy.deepcopy(model.state_dict())
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
            # sign_opr[sign_opr == 0] = 1
            # print(name, tau, torch.sum(torch.abs(sign_opr)), torch.numel(sign_opr))
            temp_dict[name] += gap_opr * sign_opr / 2
            temp_dict2[name] -= gap_opr * sign_opr / 2

        # if mask is 0, then alpha is 0, that means we do not mix
        oness = torch.ones(dictShape).cuda(dev)
        if name in mask:
            alpha = torch.rand(dictShape).cuda(dev) * mask[name]
        else:
            alpha = torch.rand(dictShape).cuda(dev)
        temp_dict[name] = (oness - alpha) * temp_dict[name] + alpha * temp_dict2[name]
    model.load_state_dict(temp_dict)

def model_momemtum_mix(model, model2, tau, mask={}, train_fc_only=False):
    dev = next(model.parameters()).device
    dev2 = next(model2.parameters()).device
    if dev != dev2:
        print("model1 and model2 are not on the same cuda device!")
        assert (False)
    temp_dict = copy.deepcopy(model.state_dict())
    temp_dict2 = model2.state_dict()
    for name, param in model.named_parameters():
        if train_fc_only and "linear" not in name:
            continue
        dictShape = temp_dict[name].shape
        gap_opr = torch.abs(temp_dict[name] - temp_dict2[name])
        gap_opr = torch.clamp(gap_opr, min=0, max=tau)
        #gap_opr = torch.add(-gap_opr, tau)
        #gap_opr = 0
        sign_opr = torch.sign(temp_dict[name] - temp_dict2[name])
        sign_opr[sign_opr == 0] = 1
        # print(name, tau, torch.sum(torch.abs(sign_opr)), torch.numel(sign_opr))
        # temp_dict[name] = temp_dict[name] + gap_opr * sign_opr / 2
        # temp_dict2[name] = temp_dict2[name] - gap_opr * sign_opr / 2

        # if mask is 0, then alpha is 0, that means we do not mix
        oness = torch.ones(dictShape).cuda(dev)
        if name in mask:
            alpha = torch.rand(dictShape).cuda(dev) * mask[name]
        else:
            alpha = torch.rand(dictShape).cuda(dev)
        #temp_dict[name] = (oness - alpha) * temp_dict[name] + alpha * temp_dict2[name] + temp_dict_0[name]
        #temp_dict[name] = (temp_dict[name] - temp_dict2[name]) + temp_dict[name]
        temp_dict[name] += alpha * gap_opr * sign_opr

    model.load_state_dict(temp_dict)


def dot_product(model, true_grad):
    dev = next(model.parameters()).device
    product = torch.tensor(0.0).cuda(dev)
    true_norm = torch.tensor(0.0).cuda(dev)
    store_norm = torch.tensor(0.0).cuda(dev)
    for name, param in model.named_parameters():
        if name not in true_grad:
            continue
        product += torch.sum(param.grad * true_grad[name])
        norms = true_grad[name].norm(p=2)
        true_norm += norms * norms
        norms = param.grad.norm(p=2)
        store_norm += norms * norms
    return product / torch.sqrt(true_norm * store_norm)


# compute gradient for batchnorm layers using input and target and set it to model.grad
def recompute_bn_gradient(model, store_model, criterion, input_group, target_group, sample_size, times,
                          use_prune, use_trunc, use_norm, prune_percentage, max_val, clip_norm):
    assert (target.size()[0] >= sample_size * times)
    assert (input.size()[0] >= sample_size * times)

    bn_store_norm = {}
    copy_model(store_model, model)
    for j in range(times):
        new_input = input_group[j * sample_size: (j + 1) * sample_size]
        new_target = target_group[j * sample_size: (j + 1) * sample_size].long()
        store_output = store_model(new_input)
        store_loss = criterion(store_output, new_target)
        store_model.zero_grad()
        store_loss.backward()

        if use_prune:
            prune_grad_percentage(store_model, prune_percentage)
        if use_trunc:
            trunc_grad(store_model, max_val)
        if use_norm:
            normalize_grad(store_model, clip_norm)

        for name, param in store_model.named_parameters():
            if "bn" not in name:
                continue
            if name not in bn_store_norm:
                bn_store_norm[name] = param.grad / times
            else:
                bn_store_norm[name] += param.grad / times

        for name, param in model.named_parameters():
            if "bn" in name:
                param.grad = bn_store_norm[name]


def copy_model(model, model2):
    model.load_state_dict(copy.deepcopy(model2.state_dict()))





