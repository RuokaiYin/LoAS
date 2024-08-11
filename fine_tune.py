import os
import time
import torch
import utils
import config_ft
import torchvision
import torch.nn as nn
import numpy as np
from statistics import mean
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn

#! The network architecture
from archs.cifarsvhn.vgg import vgg16_bn
from archs.cifarsvhn.resnet import ResNet19
from archs.cifarsvhn.alexnet import AlexNet


from utils_for_snn_lth import *
from utils import data_transforms
import copy
import torchvision.transforms as transforms
from spikingjelly.clock_driven.functional import reset_net
import pickle


args = config_ft.get_args()
def main():
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)

    train_transform, valid_transform = data_transforms(args)

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10


    #! If you have your own model, you wanna bring them here
    if args.arch == 'vgg16':
        model = vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
        model.load_state_dict(torch.load('./sample_ckpts/vgg16_final_dict.pth.tar').state_dict())
    elif args.arch == 'resnet19':
        model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
        model.load_state_dict(torch.load('./sample_ckpts/resnet19_final_dict.pth.tar').state_dict())
    elif args.arch == 'alexnet':
        model = AlexNet(num_classes=n_class, total_timestep=args.timestep).cuda()
        model.load_state_dict(torch.load('./sample_ckpts/alexnet_final_dict.pth.tar'))


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

    
    criterion = nn.CrossEntropyLoss()


    _SP_PROFILE_ = False
    _SAVE_CKPT_ = False
    _REPRODUCE_ = args.artifact

    #! Turn on the profiling
    if _SP_PROFILE_:
        n_layer = 0
        hook_fn.results = {}
        hook_fn.weighted_layers = {}
        hook_fn.weight_sparsity = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if n_layer == 0:
                    print('skip for thre first layer.')
                    n_layer+=1
                    continue
                print(f"Register hook function for Conv Layer{n_layer}.")
                hook = module.register_forward_hook(hook_fn(n_layer, args))
                hook_fn.results[n_layer] = [0]*(args.timestep+1)
                n_layer+=1

    target_accuracy = test(model, val_loader, criterion, 0)
    print(f'The original accuracy without mask is: {target_accuracy}')
    polluted_accuracy = test(model, val_loader, criterion, args.n_mask)
    print(f'The original accuracy with mask is: {polluted_accuracy}')
    if _REPRODUCE_:
        #! Create the plotting list for reproduce the Fig.11 in the paper for artifact evaluation.
        plot_array = [target_accuracy, polluted_accuracy]

    best_accuracy = 0
    for epoch_ in range(args.epoch):
        loss = 0
        accuracy = 0
        
        loss = train(model, train_loader, criterion, optimizer, scheduler, args.n_mask)
        accuracy = test(model, val_loader, criterion, args.n_mask)
        comp1 = utils.print_nonzeros(model) #this shows weight sparsity!

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if _SAVE_CKPT_:
                checkdir(f"{os.getcwd()}/finetune/{args.arch}/{args.dataset}/mask{args.n_mask}")
                torch.save(model, f"{os.getcwd()}/finetune/{args.arch}/{args.dataset}/mask{args.n_mask}/final_dict.pth.tar")
            
        print(f'Fine-tune Epoch: {epoch_}/{args.epoch} Weight Sparsity: {100-comp1:.3f} Loss: {loss:.6f} Accuracy: {accuracy:.3f}% Best Accuracy: {best_accuracy:.3f}%')
        if _REPRODUCE_:
            if epoch_ == 0:
                plot_array.append(best_accuracy)
            elif epoch_ == 4:
                plot_array.append(best_accuracy)
            elif epoch_ == 9:
                plot_array.append(best_accuracy)
                with open("./artifact/Fig-11/FT_artifact.txt", "a") as text_file:
                    for x in plot_array:
                        text_file.write(f'{x},')
                    text_file.write(f'\n')
                break
        
    
    if _SP_PROFILE_:
        print("------Profiling the silent neuron sparsity------")
        network_sparsity_silent = 0.0
        total_size = 0.0
        for layer_name, result in hook_fn.weighted_layers.items():
            total_size += result
        for layer_name, result in hook_fn.results.items():
            num_samples = len(val_loader)
            profile_str = f"Average Percentage for spikes on Layer {layer_name}: "
            for i in range(args.timestep+1):
                avg_spk = result[i] / num_samples
                profile_str += (f"[{i} spikes {round(avg_spk*100,2)}] ")
                if i == 0:
                    network_sparsity_silent += (hook_fn.weighted_layers[layer_name]/total_size)*avg_spk 
            print(profile_str)
        print(f"Weight Spikes Sparsity Across Layers {round(network_sparsity_silent*100,2)}")



def test(model, test_loader, criterion, n_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in (test_loader):
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            if args.arch == 'resnet19':
                output = sum(model((data, n_mask)))
            else:
                output = sum(model(data, n_mask)) #! mask is sending in here to filter out the neurons with low firing activity.
            reset_net(model)
            _,idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            
            correct += idx.eq(target.data.view_as(idx)).sum().item()      
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy




def train(model, train_loader, criterion, optimizer, scheduler, n_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    EPS = 1e-6
    
    for batch_idx, (imgs, targets) in (enumerate(tqdm(train_loader))):
        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        with amp.autocast():
            if args.arch == 'resnet19':
                output = model((imgs, n_mask))
            else:
                output = model(imgs, n_mask)
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
