import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
nc =10
def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x


def clip_data(data, max_norm):
    norms = torch.norm(data.reshape(data.shape[0], -1), dim=-1)
    scale = (max_norm / norms).clamp(max=1.0)
    data *= scale.reshape(-1, 1, 1, 1)
    return data


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats):
        super(StandardizeLayer, self).__init__()
        self.bn_stats = bn_stats

    def forward(self, x):
        return standardize(x, self.bn_stats)


class ClipLayer(nn.Module):
    def __init__(self, max_norm):
        super(ClipLayer, self).__init__()
        self.max_norm = max_norm

    def forward(self, x):
        return clip_data(x, self.max_norm)


class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                #cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']
            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
                #cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ScatterLinear(nn.Module):
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=nc):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


def resnet01():
    return CNN()



CNNS = {
    "cifar10": CIFAR10_CNN,
    #"cifar10": resnet01,
    "fmnist": MNIST_CNN,
    "mnist": MNIST_CNN,
}



