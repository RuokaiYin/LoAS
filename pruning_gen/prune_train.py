import os
import time
import copy
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
from statistics import mean
from spikingjelly.clock_driven.functional import reset_net

from archs.cifarsvhn.resnet import ResNet19

import config_lth
from utils_for_snn_lth import *


args = config_lth.get_args()

def main():
    cudnn.benchmark = True
    cudnn.deterministic = True
    args = config_lth.get_args()
    train_transform, valid_transform = data_transforms(args)

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10

    criterion = nn.CrossEntropyLoss()

    if args.arch == 'resnet19':
        model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.5),int(args.epoch*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epoch), eta_min= 0)
    else:
        print ("will be added...")
        exit()



    #! Pruning
    # NOTE First Pruning Iteration is of No Compression
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros(args.epoch, float)
    all_accuracy = np.zeros(args.epoch, float)
    rewinding_epoch = 0

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    checkdir(f"{os.getcwd()}/lth/{args.arch}/{args.dataset}")
    torch.save(model.state_dict(), f"{os.getcwd()}/lth/{args.arch}/{args.dataset}/initial_state_dict.pth.tar")

    # Making Initial Mask
    mask = make_mask(model)

    for round_ in range(ITERATION):
        best_accuracy = 0

        if not round_ == 0:
            # Dumping mask _before
            checkdir(f"{os.getcwd()}/lth/{args.arch}/{args.dataset}/mask")
            with open(f"{os.getcwd()}/lth/{args.arch}/{args.dataset}/mask/{round_}_mask_{comp1}.pkl",'wb') as fp:
                pickle.dump(mask, fp)

            model, mask = prune_by_percentile(args, args.prune_percent, mask , model)
            model = original_initialization(mask, initial_state_dict, model)

            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
            else:
                print ("will be added...")
                exit()

            if args.scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.5),int(args.epoch*0.75)], gamma=0.1)
            elif args.scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epoch), eta_min= 0)
            else:
                print ("will be added...")
                exit()


        print(f"\n--- Pruning Level [round{args.round}:{round_}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model)
        comp[round_] = comp1
        loss = 0
        accuracy =0
        for iter_ in range(args.epoch - rewinding_epoch):
            if (iter_+1) % args.valid_freq == 0:
                accuracy = test(model, val_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    checkdir(f"{os.getcwd()}/lth/{args.arch}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/lth/{args.arch}/{args.dataset}/{round_}_model.pth.tar")

            # Training
            loss = train(model, train_loader, criterion, optimizer, scheduler)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            #TODO Late rewinding init weight at 20epoch
            if round_ == 0 and iter_ == args.rewinding_epoch:
                print ('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                rewinding_epoch = args.rewinding_epoch

            # Frequency for Printing Accuracy and Loss
            if (iter_ +1)% args.print_freq == 0:
                print(f'Train Epoch: {iter_}/{args.epoch} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        bestacc[round_]=best_accuracy
        print("Round best accuracy: ", best_accuracy)



def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = sum(model(data))
            reset_net(model)
            _,idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            
            correct += idx.eq(target.data.view_as(idx)).sum().item()      
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def train(model, train_loader, criterion, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    EPS = 1e-6
    
    for batch_idx, (imgs, targets) in (enumerate(tqdm(train_loader))):
        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        with amp.autocast():
            output = model(imgs)
            train_loss = sum([criterion(s, targets) for s in output]) / args.timestep
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if (len(tensor.size())) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
    scheduler.step()
    return train_loss.item()

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
if __name__ == '__main__':
    main()
