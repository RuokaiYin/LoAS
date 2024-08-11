import torch
import time
import math
import torch.nn.functional as F
from tqdm import tqdm
import argparse

#! Global counters
total_numel = 0
total_nz = 0
total_sp = 0
total_sp_numel = 0
total_w_nz = 0
total_w_numel = 0

def loas_spike_pack(spike_mat, mode='normal'):

    assert len(spike_mat.shape)==3, "spike tensor should have 3 dim!"
    global total_numel
    global total_nz

    #! This is printing the spikes/neuron for this layer, BEFORE the packin.
    print(f'Spikes/Neuron: {round((1-torch.count_nonzero(spike_mat)/spike_mat.numel()).item()*100,4)}%')
    
    pack_spike_mat = spike_mat.sum(dim=0) #! Simply sum the T dimension, equivalent to pack the spikes

    #! This is printing the silent neuron ratio, AFTER the packin.
    print(f'Ratio of Silent Neurons: {round((1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel()).item()*100,4)}%')
    
    #! Please refer to the paper, the strong mode is esentially further consider the neuron only spikes 1 time also as silent neuron.
    if mode == 'strong':
        #! We simply further mask out the neuron that spikes 1 time.
        pack_spike_mat = pack_spike_mat*((pack_spike_mat!=1.0).float()) 
        #TODO: The users can add any more masks here for experiments in the format of "pack_spike_mat = pack_spike_mat*((pack_spike_mat!=t).float())", 
        #TODO: where t is the threshold of # of spikes for masking out.

        print(f"In the strong LoAS mode: Strong silent neuron ratios: {round((1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel()).item()*100,4)}")

    #! Update the global counter for the global silent neuron sparsity data of the entire SNN.
    total_numel += pack_spike_mat.numel()
    total_nz += torch.count_nonzero(pack_spike_mat)
    
    return pack_spike_mat.unsqueeze(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-simulation")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--loas', type=str, default='normal', help='[normal, strong]')
    parser.add_argument('--store', action='store_true', help='Usin this key to store the post-processed matrices.')
    parser.add_argument('--artifact', action='store_true', help='Usin this key to turn on the artifact mode.')
    args = parser.parse_args()

    PATH = f'./src_matrices/{args.arch}_final_matrices_dict.pth' #! Path where the matrices from the PyTorch model is stored
    
    #* Have been tested on Apple M1-Pro
    matrices_dict = torch.load(PATH,map_location=torch.device('cpu')) 

    loas_matrices_dict = {}
    mode = args.loas
    artifact_str = "" #! For sotring the result string only for artifact purpose.

    #! Tranverse the PyTorch-based matrices to get the layerwise info.
    for k,v in matrices_dict.items():
        loas_matrices_dict[k]={}
        
        inp_mat = v['x'] #! Here is the spike matrix, with shape of (T.M,K), batch is set to 1 for inference.
        weight_mat = v['w'] #! Here is the weight matrix
        
        #! This is for calculatin the original spike sparisty (aka, spikes/neuron, as defined in other works)
        total_sp += torch.count_nonzero(inp_mat)
        total_sp_numel += inp_mat.numel() #! We count the number of neurons across the network for normalizing at the end.

        print(k) #! Print the name of the layer
        #! Main function to pack the spike, it also print out the spike info.
        loas_spike_mat = loas_spike_pack(inp_mat,mode) 
        
        print(f'Weight sparsity of the layer: {round((1-torch.count_nonzero(weight_mat)/weight_mat.numel()).item()*100,4)}%')
        total_w_nz += torch.count_nonzero(weight_mat)
        total_w_numel += weight_mat.numel()
        #! Below is for storing purpose only.
        loas_matrices_dict[k]['x'] = loas_spike_mat
        loas_matrices_dict[k]['w'] = v['w']
        #! Here is for artifact purpose only.
        if args.artifact:
            #? We always have one less layer count here, since we are not considering the direct encoding layer (1st layer in the network)
            if (args.arch == 'vgg16' and k == 'Layer_7') or (args.arch == 'resnet19' and k == 'Layer_18') or (args.arch == 'alexnet' and k == 'Layer_3'):
                artifact_str += f'Layerwise result of {args.arch} with mode {mode} show below: \n'
                artifact_str += k
                artifact_str += '\n'
                artifact_str += f'Spikes/Neuron: {round((1-torch.count_nonzero(inp_mat)/inp_mat.numel()).item()*100,4)}%\n'
                pack_spike_mat = inp_mat.sum(dim=0)
                if mode == 'normal':
                    artifact_str += f'Ratio of Silent Neurons: {round((1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel()).item()*100,4)}%\n'
                else:
                    pack_spike_mat = pack_spike_mat*((pack_spike_mat!=1.0).float())
                    artifact_str += f"In the strong LoAS mode: Strong silent neuron ratios: {round((1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel()).item()*100,4)}%\n"
                artifact_str += f'Weight sparsity of the layer: {round((1-torch.count_nonzero(weight_mat)/weight_mat.numel()).item()*100,4)}%\n'

    print("---- Successfully convert the PyTorch model into LoAS-packed Matrices. ----")
    print(f'Final Network Statics is printed out below: ')
    print(f"On {args.arch}, original spike sparsity: {round((1-(total_sp/total_sp_numel)).item()*100,4)}%")
    print(f"On {args.arch}, under mode of {mode}, packed spike sparsity: {round((1-(total_nz/total_numel)).item()*100,4)}%")
    print(f"On {args.arch}, the weight sparsity: {round((1-(total_w_nz/total_w_numel)).item()*100,4)}%")

    if args.artifact:
        artifact_str += (f"On {args.arch}, original spike sparsity: {round((1-(total_sp/total_sp_numel)).item()*100,4)}%\n")
        artifact_str += (f"On {args.arch}, under mode of {mode}, packed spike sparsity: {round((1-(total_nz/total_numel)).item()*100,4)}%\n")
        artifact_str += (f"On {args.arch}, the weight sparsity: {round((1-(total_w_nz/total_w_numel)).item()*100,4)}%\n")
        with open("./tableII_artifact.txt", "a") as text_file:
            text_file.write(artifact_str)
            text_file.write('\n')


    if args.store:
        torch.save(loas_matrices_dict, f'./{args.arch}_final_matrices_dict_loas_{mode}.pth')
