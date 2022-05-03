import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, model2, train_loader, optimizer, optimizer2, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if math.floor(batch_idx / n_acc_steps) % 2 == 0:
            temp_model = model
            temp_model2 = model2
            temp_optimizer = optimizer
        else:
            temp_model = model2
            temp_model2 = model
            temp_optimizer = optimizer2

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)

        output = temp_model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):

            temp_dict = temp_model.state_dict()
            temp_dict2 = temp_model2.state_dict()
            for j in temp_dict.keys():
                gap_opr = torch.abs(temp_dict[j] - temp_dict2[j])
                if (torch.min(gap_opr) < tau):
                    gap_opr = torch.clamp(gap_opr, 0, tau)
                    gap_opr = torch.add(-gap_opr, tau)
                    sign_opr = torch.sign(temp_dict[j] - temp_dict2[j])
                    sign_opr[sign_opr == 0] = 1
                    temp_dict[j] = torch.add(temp_dict[j], gap_opr * sign_opr / 2)
                    temp_dict2[j] = torch.add(temp_dict2[j], -gap_opr * sign_opr / 2)

                dictShape = temp_dict[j].shape
                alpha1 = torch.rand(dictShape).cuda(device)
                temp_dict[j] = alpha1 * temp_dict[j] + (torch.ones(dictShape).cuda(device) - alpha1) * temp_dict2[j]
            temp_model.load_state_dict(temp_dict)
            
            temp_optimizer.step()
            temp_optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                temp_optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc
