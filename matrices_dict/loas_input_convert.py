import torch
import time
import math
import torch.nn.functional as F
from tqdm import tqdm
import argparse

total_numel = 0
total_nz = 0
total_sp = 0
total_sp_numel = 0

def loas_spike_pack(spike_mat, mode='normal'):

    assert len(spike_mat.shape)==3, "spike tensor should have 3 dim!"
    global total_numel
    global total_nz

    print(1-torch.count_nonzero(spike_mat)/spike_mat.numel())
    pack_spike_mat = spike_mat.sum(dim=0)
    print(1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel())
    

    if mode == 'strong':
        pack_spike_mat = pack_spike_mat*((pack_spike_mat!=1.0).float())
        print(1-torch.count_nonzero(pack_spike_mat)/pack_spike_mat.numel())
        # print('matrice after strong pack \n', pack_spike_mat)

    total_numel += pack_spike_mat.numel()
    total_nz += torch.count_nonzero(pack_spike_mat)
    return pack_spike_mat.unsqueeze(dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-simulation")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--loas', type=str, default='normal', help='[normal, strong]')
    args = parser.parse_args()

    path = f'./matrices_dict/{args.arch}_final_matrices_dict.pth'
    matrices_dict = torch.load(path,map_location=torch.device('cpu'))

    loas_matrices_dict = {}
    mode = args.loas

    for k,v in matrices_dict.items():
        loas_matrices_dict[k]={}
        inp_mat = v['x']
        print(inp_mat.shape)
        print(v['w'].shape)
        total_sp += torch.count_nonzero(inp_mat)
        total_sp_numel += inp_mat.numel()

        loas_spike_mat = loas_spike_pack(inp_mat,mode)
        
        print(k)
        # print(1-torch.count_nonzero(inp_mat)/inp_mat.numel())
        loas_matrices_dict[k]['x'] = loas_spike_mat
        loas_matrices_dict[k]['w'] = v['w']

    # torch.save(loas_matrices_dict, f'./matrices_dict/{args.arch}_final_matrices_dict_loas_{mode}.pth')
    print("---- Successfully convert the PyTorch model into LoAS-packed Matrices. ----")
    print(f"On {args.arch}, original spike sparsity: {1-(total_sp/total_sp_numel)}")
    print(f"On {args.arch}, packed spike sparsity: {1-(total_nz/total_numel)}")