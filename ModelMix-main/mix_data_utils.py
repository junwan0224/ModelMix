import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def mixup_data(x, y, alpha=0.1):
    dev = x.device
    if alpha > 0:
        # lam = np.random.beta(alpha, alpha)
        lam = 0.1
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(dev)
    index0 = torch.randperm(batch_size).cuda(dev)
    mixed_x = (1 - 2 * lam) * x + lam * x[index, :] + lam * x[index0, :]
    y_a, y_b, y_c = y, y[index], y[index0]
    return mixed_x, y_a, y_b, y_c, lam


def mixup_public_data(private_x, private_y, public_x, public_y, index, index0, public_batch_size, batch_select, alpha=0.1):
    if alpha > 0:
        # lam = np.random.beta(alpha, alpha)
        lam = 0.1
    else:
        lam = 1

    batch_size = public_x.size()[0]
    repeat_num = int(public_batch_size / batch_select)
    # index = torch.randint(batch_size, (public_batch_size,)).cuda(device)
    x2, y2 = public_x[index], public_y[index]
    # index0 = torch.randint(batch_size, (public_batch_size,)).cuda(device)
    x3, y3 = public_x[index0], public_y[index0]
    x1, y1 = private_x.repeat(repeat_num, 1, 1, 1), private_y.repeat(repeat_num)
    # print(x1.size(), x2.size(), x3.size())
    mixed_x = (1 - 2 * lam) * x1 + lam * x2 + lam * x3
    return mixed_x, y1, y2, y3, lam


def mixup_criterion(criterion, pred, y_a, y_b, y_c, lam):
    return (1 - 2 * lam) * criterion(pred, y_a) + lam * criterion(pred, y_b) + lam * criterion(pred, y_c)