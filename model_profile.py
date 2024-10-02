import os
import time
import torch
import utils
import pickle
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from spikingjelly.clock_driven.functional import reset_net

#! The network architecture
from archs.cifarsvhn.vgg import vgg16_bn
from archs.cifarsvhn.resnet import ResNet19
from archs.cifarsvhn.alexnet import AlexNet

#! The helper functions
import utils_for_snn_lth
import config_profile
from utils import data_transforms, hook_fn

#! Global args
args = config_profile.get_args()

def main():
    print("------------ LoAS inference profiling for dual-sparse SNNs ----------")
    print('The profiling args are below:')
    print(args)

    train_transform, valid_transform = data_transforms(args)

    #! Feel free to extend this to new datasets
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=valid_transform)
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

    _SP_PROFILE_ = args.profile

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
        
        n_layer = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                tensor = module.weight.data.cpu().numpy()
                nz_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                print(f"Weight Sparsity on Layer: {100-round((nz_count/total_params)*100,1)}")
                n_layer+=1
    
    comp1 = utils.print_nonzeros(model) #this shows weight sparsity!
    t1 = time.time()
    accuracy= test(model, val_loader)
    t2 = time.time()
    print(f"Time used for inference: {round(t2-t1,3)}s")
    print("Accuracy: ", accuracy)
    print("Weight Sparsity: ", 100-comp1)

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
        

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            batch = data.shape[0]
            data, target = data.to(device), target.to(device)
            if args.arch == 'resnet19':
                output = sum(model((data, args.n_mask)))
            else:
                output = sum(model(data, args.n_mask)) #! mask is sending in here to filter out the neurons with low firing activity.
            reset_net(model)
            _,idx = output.data.max(1, keepdim=True)  # get the index of the max log-probability
            
            correct += idx.eq(target.data.view_as(idx)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__ == '__main__':
    main()
