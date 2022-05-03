import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torchvision.utils as vutils
import torch.nn.init as init

def make_mask(model):
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            mask[step] = torch.ones_like(param.data)
            step = step + 1
    return mask


def mask_initial (mask, model, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = mask[step] * param.data + (torch.ones_like(mask[step]) - mask[step]) * initial_state_dict[name]
            step = step + 1


def prune_by_percentile(percent, mask, model, dev):
    all_param = torch.empty(0).cuda(dev)
    for name, param in model.named_parameters():
        # We do not prune bias term
        dev = param.device
        if 'weight' in name:
            #print(all_param.device, torch.flatten(param.data).device)
            all_param = torch.cat((all_param, torch.flatten(param.data)), 0)
    percentile_value = np.percentile(abs(all_param.cpu().numpy()), percent)
            
    step = 0
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            new_mask = np.where(abs(tensor) < percentile_value, 0, 1)
            mask[step] = torch.from_numpy(new_mask).cuda(dev)
            step = step + 1


def mask_grad (model, mask):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.grad = torch.from_numpy(param.grad.cpu().numpy() * mask[step])
            step = step + 1


