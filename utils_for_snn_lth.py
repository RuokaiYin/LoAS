import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import copy

from spikingjelly.clock_driven.functional import reset_net
import math

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

def layerwise_utilization_check(mask,layer_index, n_PE,arch_style='systolic'):
    layer = mask[layer_index]
    # print(len(mask))
    # print(layer_index)
    # print(layer.shape)
    utilization_list = []
    dynamic_cnts = 0
    wasted_cnts = 0
    latency = 0

    # print(layer.shape)

    if arch_style == 'systolic':
        PE_load = math.ceil(layer.shape[1]/n_PE) ### Input channel divide the number of PEs
        if PE_load < 1:
            print("The layer's input channel is smaller than number of PEs of INDEX:",layer_index)
            return None
        for i in range (layer.shape[0]): ### Loop through the output channel
            channel_load_list = []
            idx = 0
            
            for j in range (n_PE):

                workload = torch.count_nonzero(torch.Tensor(layer[i,idx:(idx+PE_load)]))
                channel_load_list.append(workload.data.cpu())
                dynamic_cnts += workload.data.cpu()
                idx = idx + PE_load

            if max(channel_load_list) == 0:
                print("At layer:",layer_index,"output_channel:",i)
                continue
            else:
                # idle_cycle_count +=  max(channel_load_list) - min(channel_load_list)

                # utilization = min(channel_load_list)/max(channel_load_list)
                avg_workload = math.ceil(sum(channel_load_list)/len(channel_load_list))
                utilization = 1 - ((max(channel_load_list) -avg_workload)/max(channel_load_list))*(n_PE/(n_PE-1))
                utilization_list.append(utilization)
                wasted_list = [max(channel_load_list) - i for i in channel_load_list]
                wasted_cnts += sum(wasted_list)
                latency += max(channel_load_list)

    elif arch_style == 'sata':
        ### assign a whole output channel to PE

        for i in range (math.ceil(layer.shape[0]/n_PE)): ## Assign the whole output channels to PE
            channel_load_list = []
            i = i*n_PE

            for j in range (n_PE):
                workload = torch.count_nonzero(torch.Tensor(layer[i+j]))
                channel_load_list.append(workload.data.cpu())
                dynamic_cnts += workload.data.cpu()
            if max(channel_load_list) == 0:
                # print("At layer:",layer_index,"output_channel:",i)
                continue
            else:
                # utilization = min(channel_load_list)/max(channel_load_list)
                avg_workload = math.ceil(sum(channel_load_list)/len(channel_load_list))
                utilization = 1 - ((max(channel_load_list) -avg_workload)/max(channel_load_list))*(n_PE/(n_PE-1))
                utilization_list.append(utilization)
                wasted_list = [max(channel_load_list) - i for i in channel_load_list]
                wasted_cnts += sum(wasted_list)
                latency += max(channel_load_list)
    else:
        print("Arch config is not supported.")
        exit()

    
    avg_utilization = sum(utilization_list)/len(utilization_list)
    max_utilization = max(utilization_list)
    min_utilization = min(utilization_list)


    return avg_utilization, max_utilization, min_utilization, dynamic_cnts, wasted_cnts, latency

def recover_weight_utilization(mask,layer_index,channel_index,pe_load,avg_workload,workload_list,arch_style='systolic',mode='strong',model=None):
    
    new_work_load_list = []
    
    if arch_style == 'systolic':
        idx = 0
        for p in workload_list:
            if p < avg_workload:
                
                workload_gap = avg_workload - p
                recover_channel = torch.Tensor(mask[layer_index][channel_index][idx:idx+pe_load])

                temp_list = (recover_channel==0).nonzero()
                # print(temp_list.shape)
                if mode == "grad":
                    # model[]
                    temp_list
                else:
                    recover_idx = torch.randperm(temp_list.shape[0])
                    recover_idx = recover_idx[:workload_gap]
                superidx = torch.transpose(temp_list[recover_idx],0,1)
                recover_idx = superidx.tolist()

                # new_zero_mask = torch.zeros_like(recover_channel)
                # print(recover_idx.shape)
                recover_channel[recover_idx] = 1
                new_work_load_list.append(recover_channel)
            elif mode == 'strong' and p > avg_workload:
                workload_gap = p - avg_workload
                recover_channel = torch.Tensor(mask[layer_index][channel_index][idx:idx+pe_load])

                temp_list = (recover_channel==1).nonzero()
                reduce_idx = torch.randperm(temp_list.shape[0])
                reduce_idx = reduce_idx[:workload_gap]
                superidx = torch.transpose(temp_list[reduce_idx],0,1)
                reduce_idx = superidx.tolist()

                recover_channel[reduce_idx] = 0
                new_work_load_list.append(recover_channel)
            else:
                origin_workload = torch.Tensor(mask[layer_index][channel_index][idx:idx+pe_load])
                new_work_load_list.append(origin_workload)
            idx += pe_load
    elif arch_style == 'sata':
        idx = 0
        # print(model.shape)
        for p in workload_list:
            # print(p)
            # print(workload_list)
            if p < avg_workload:
                # print("a")
                workload_gap = avg_workload - p
                recover_channel = torch.Tensor(mask[layer_index][idx+channel_index]).cpu() ## Take the entire output channel

                temp_list = (recover_channel==0).nonzero()
                # print("t",temp_list.shape)
                if mode == 'grad':
                    temp_w = model[idx+channel_index].cpu()
                    temp_try = torch.transpose(temp_list,0,1).tolist()
                    a,i = torch.sort(temp_w[temp_try],descending=True)
                    fakeidx = torch.transpose(temp_list[i[:workload_gap]],0,1)
                    recover_idx = fakeidx.tolist()
                else:
                    recover_idx = torch.randperm(temp_list.shape[0])
                    recover_idx = recover_idx[:workload_gap]
                    # print(recover_idx.shape)
                    superidx = torch.transpose(temp_list[recover_idx],0,1)
                    recover_idx = superidx.tolist()

                recover_channel[recover_idx] = 1
                new_work_load_list.append(recover_channel)
            elif mode == 'strong' and p > avg_workload:
                # print("b")
                workload_gap = p - avg_workload
                recover_channel = torch.Tensor(mask[layer_index][idx+channel_index])

                temp_list = (recover_channel==1).nonzero()
                # print(temp_list.shape)
                reduce_idx = torch.randperm(temp_list.shape[0])
                reduce_idx = reduce_idx[:workload_gap]
                superidx = torch.transpose(temp_list[reduce_idx],0,1)
                reduce_idx = superidx.tolist()

                recover_channel[reduce_idx] = 0
                new_work_load_list.append(recover_channel)
            elif mode == 'grad' and p > avg_workload:
                workload_gap = p - avg_workload
                recover_channel = torch.Tensor(mask[layer_index][idx+channel_index])
                temp_list = (recover_channel==1).nonzero()
                temp_w = model[idx+channel_index].cpu()

                temp_try = torch.transpose(temp_list,0,1).tolist()
                a,i = torch.sort(temp_w[temp_try],descending=False)
                fakeidx = torch.transpose(temp_list[i[:workload_gap]],0,1)
                reduce_idx = fakeidx.tolist()
                
                recover_channel[reduce_idx] = 0
                new_work_load_list.append(recover_channel)
            else:
                # print("c")
                origin_workload = torch.Tensor(mask[layer_index][idx+channel_index])
                new_work_load_list.append(origin_workload)
            idx+=1
    else:
        print("Arch config is not supported.")
        exit()
        # print(new_work_load_list)
    return torch.stack(new_work_load_list)


def layerwise_utilization_recover(mask,layer_index,n_PE,arch_style='systolic',mode='strong',model=None):
    # print(len(mask), layer_index)
    layer = mask[layer_index]
    utilization_list = []
    weight_width = layer.shape[-1]

    if arch_style == 'systolic':
        PE_load = math.ceil(layer.shape[1]/n_PE) ### Input channel divide the number of PEs
        if PE_load < 1:
            print("The layer's input channel is smaller than number of PEs of INDEX:",layer_index)
            return None
        
        for i in range (layer.shape[0]): ### Loop through the output channel
            channel_load_list = []
            idx = 0
            
            for j in range (n_PE):
                
                workload = torch.count_nonzero(torch.Tensor(layer[i,idx:(idx+PE_load)]))
                channel_load_list.append(workload.data.cpu())
                idx = idx + PE_load

            if max(channel_load_list) == 0:
                # print("At layer:",layer_index,"output_channel:",i)
                continue
            else:
                # idle_cycle_count +=  max(channel_load_list) - min(channel_load_list)
                avg_workload = math.ceil(sum(channel_load_list)/len(channel_load_list))
                new_mask = recover_weight_utilization(mask,layer_index,i,PE_load,avg_workload,channel_load_list,arch_style,mode,model)

            new_mask = new_mask.view(-1,weight_width,weight_width)
            # print(new_mask.shape)
            mask[layer_index][i] = new_mask

    elif arch_style == 'sata':

        for i in range (math.ceil(layer.shape[0]/n_PE)): ## Assign the whole output channels to PE
            channel_load_list = []
            i = i*n_PE

            for j in range (n_PE):
                workload = torch.count_nonzero(torch.Tensor(layer[i+j]))
                channel_load_list.append(workload.data.cpu())
            if max(channel_load_list) == 0:
                # print("At layer:",layer_index,"output_channel:",i)
                continue
            else:
                
                avg_workload = math.ceil(sum(channel_load_list)/len(channel_load_list))
                new_mask = recover_weight_utilization(mask,layer_index,i,0,avg_workload,channel_load_list,arch_style,mode,model)
            
            new_mask = new_mask.view(-1,layer.shape[1],weight_width,weight_width)
            mask[layer_index][i:i+n_PE] = new_mask
    else:
        print("Arch config is not supported.")
        exit()

        
    return mask

def utilization_operation_network(mask,model,n_PE,operation="check",arch_style='systolic',mode='strong'):
    
    mask_idx = 0
    grad_i = 0
    layerwise_u_list_avg = []
    layerwise_u_list_max = []
    layerwise_u_list_min = []
    layerwise_d_cycles = []
    layerwise_wasted_cycles = []
    layerwise_latency_cycles = []

    layer_list = []
    for name, param in model.named_parameters():
        # print(param.shape)
        if len(param.shape) == 4:
            layer_list.append(param)
    layer_list.pop(0)

    for name, module in model.named_modules():
        
        # if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if isinstance(module, nn.Conv2d):
            if mask_idx == 0:
                mask_idx += 2
                continue
            # print(module)
            # print(mask_idx)
            # print(mask[mask_idx].shape)
            if operation == "check":
                avg_util, max_util, min_util, dyna, wasted, latency = layerwise_utilization_check(mask,mask_idx, n_PE,arch_style)
                layerwise_u_list_avg.append(avg_util)
                layerwise_u_list_max.append(max_util)
                layerwise_u_list_min.append(min_util)
                layerwise_d_cycles.append(dyna)
                layerwise_wasted_cycles.append(wasted)
                layerwise_latency_cycles.append(latency)

            elif operation == "recover":
                # print(len(layer_list))
                # print(mask_idx)
                # print(grad_i)
                mask = layerwise_utilization_recover(mask,mask_idx, n_PE,arch_style,mode,layer_list[grad_i])
                grad_i+=1
            
            mask_idx += 2 ### Hardcode the mask index for bn-based vgg
            if mask_idx > len(mask):
                break
    if operation == "check":
        return layerwise_u_list_avg, layerwise_u_list_max, layerwise_u_list_min, layerwise_d_cycles,layerwise_wasted_cycles,layerwise_latency_cycles
    else:
        return mask


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


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def computing_firerate(module, inp, out):

    fired_spikes = torch.count_nonzero(out)
    module.spikerate += fired_spikes/8.0
    module.num_neuron += np.prod(out.shape[1:len(out.shape)])/8.0

def test_spa(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    #TODO
    overall_nueron =0
    overall_spike = 0

    ### Defining Sparsity Handling Code
    neuron_type = 'LIFNode'
    for name, module in model.named_modules():
        if neuron_type in str(type(module)):
            module.register_forward_hook(computing_firerate)
            module.spikerate = 0
            module.num_neuron = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch = data.shape[0]

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    for name, module in model.named_modules():
        if neuron_type in str(type(module)):
            overall_nueron += module.num_neuron/len(test_loader)
            overall_spike += module.spikerate/len(test_loader.dataset)
            # print(overall_nueron)
            # print(module.spikerate/len(test_loader.dataset))
            # print(module.spikerate)
    print("Overall spike rate:", overall_spike/overall_nueron)


    return accuracy,overall_spike/overall_nueron

def test_ann(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy




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

