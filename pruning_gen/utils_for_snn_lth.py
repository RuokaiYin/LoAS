import numpy as np
import random
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def data_transforms(args):
    if args.dataset == 'fmnist':
        MEAN = [0.5]
        STD = [0.5]
    elif  args.dataset == 'svhn':
        MEAN = [0.5,0.5,0.5]
        STD = [0.5,0.5,0.5]
    elif args.dataset == 'cifar10':
        MEAN = [0.4913, 0.4821, 0.4465]
        STD = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2673, 0.2564, 0.2762]
    elif args.dataset == 'tinyimagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if  (args.dataset== 'fmnist'):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    elif  (args.dataset== 'svhn'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


    elif (args.dataset== 'tinyimagenet'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif (args.dataset == 'cifar10'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    return train_transform, valid_transform

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

    return mask

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_modules():
        if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
            tensor = p.weight.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
    print(
        f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return (round((nonzero / total) * 100, 1))

# Prune by Percentile module
def prune_by_percentile(args, percent, mask , model):

        if args.pruning_scope == 'local':
            # Calculate percentile value
            step = 0
            for name, param in model.named_parameters():

                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:
                        step += 1
                        continue
                    alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                    percentile_value = np.percentile(abs(alive), percent)

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        elif args.pruning_scope == 'global':
            step = 0
            all_param = []
            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1: # We do not prune BN term
                        continue
                    alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                    all_param.append(list(abs(alive)))
            param_whole = np.concatenate(all_param)
            percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0])/float(100./percent))]

            step = 0

            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor =  param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:  # We do not prune BN term
                        step += 1
                        continue

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        else:
            exit()

        return model, mask


def get_pruning_maks(args, percent, mask, model):
    step = 0
    all_param = []
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                continue
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            all_param.append(list(abs(alive)))
    param_whole = np.concatenate(all_param)
    percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

    step = 0

    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                step += 1
                continue
            new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
            mask[step] = new_mask
            step += 1
    step = 0

    return  mask


def original_initialization(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

    return model

def original_initialization_nobias(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name] +1

    step = 0

    return model


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

