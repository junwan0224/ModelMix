import torch
import resnet
import os
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from opacus.utils import module_modification


# keep_bn: whether to keep the batchnorm layer or not
# arch: architecture for the model
def get_model(arch, resume=None, keep_bn=True, use_r50=False):
    if use_r50:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        model2 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        # store_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    else:
        model = resnet.__dict__[arch]()
        model2 = resnet.__dict__[arch]()
        # store_model = resnet.__dict__[arch]()

    # optionally resume from a checkpoint
    if keep_bn:
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                temp_dict = model.state_dict()
                for name in temp_dict.keys():
                    if (temp_dict[name].shape != checkpoint['state_dict'][name].shape):
                        print("alert: stored model differs declared model in layer ", name)
                        checkpoint['state_dict'][name] = temp_dict[name]
                model.load_state_dict(checkpoint['state_dict'])
                model2.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(resume))
    else:
        model = module_modification.convert_batchnorm_modules(model)
        model2 = module_modification.convert_batchnorm_modules(model2)

    return (model, model2)


def get_data_SVHN():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.SVHN(root='./data', split='train', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    test_set = datasets.SVHN(root='./data', split='test', transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    return train_set, test_set


def get_data_CIFAR10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    test_set = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize]))
    return train_set, test_set


def get_data_FMINIST():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)
    test_set = datasets.FashionMNIST(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    return train_set, test_set


def get_optimizer(model, lr, momentum, decay, gamma):
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], last_epoch=args.start_epoch - 1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return optimizer, lr_scheduler


def iid_sample(dataset, q):
    inputs = torch.empty(0)
    targets = torch.empty(0)
    n = len(dataset)
    for i in range(n):
        r = random.uniform(0, 1)
        if r > q:
            continue
        else:
            img, tar = dataset.__getitem__(i)
            inputs = torch.cat((inputs, img.unsqueeze(0)), 0)
            targets = torch.cat((targets, torch.tensor(tar).unsqueeze(0)), 0)
    indexes = torch.randperm(inputs.shape[0])
    inputs = inputs[indexes]
    targets = targets[indexes]
    return inputs, targets

