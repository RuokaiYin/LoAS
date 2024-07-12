import torch
import time
import math
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import argparse


from tiling_gen import _Tile_shape_Anl_W, _Tile_shape_Anl_Inp, _Tile_to_pe_distr
from tiling_move import _Tile_movement_Anl, _Tile_movement_Custom
# from sram_profile import check_dram2sram_traffic, check_dram2sram_traffic_dense
from utils import _value2mask


#! Local function tailored to each accelerator
def _tile_to_pe_custom(inp_tile, w_tile):
    
    #! Define the number of PEs here.
    n_pe = 16
    pe_w_lists = []
    pe_x_lists = []

    #! In LoAS, the inter-PE dimensions is w.size(1)
    #! In LoAS, the inter-PE dim for x is 0, meaning all PEs shared same inputs.
    activated_pe = 0
    for i in range(n_pe):
        if i > inp_tile.size(1)-1:
            break
        pe_w_lists += [w_tile]
        pe_x_lists += [inp_tile[:,i,:]]
        activated_pe += 1
    active_ratio = activated_pe/n_pe
    
    return pe_x_lists, pe_w_lists, active_ratio

#! Local function to mimic the behavior of a single PE.
def _pe_operate(pe_x, pe_w, stats_dict):

    x = _value2mask(pe_x)
    w = _value2mask(pe_w)
    y = x.matmul(w)
    effect = torch.sum(y)

    stats_dict['tot_comps'] += effect
    latency = effect
    #! For each tile, read the wsp & non-zero weights.
    stats_dict['sram_traffic_x'] += effect
    
    return stats_dict, latency

#! Local function to mimic the behavior of the dram2sram traffic.
def _check_dram2sram_traffic(tensor, sram_buffer, sram_size_left, stats_dict, var=None):
    n = torch.count_nonzero(tensor).item()
    n_tot = tensor.numel()
    found = any(torch.equal(tensor, existing_tensor) for existing_tensor in sram_buffer)
    if found:
        stats_dict[f'hit_{var}'] += n
        return sram_buffer, sram_size_left, stats_dict
    else:
        if n > sram_size_left:
            while n > sram_size_left:
                sram_size_left += torch.count_nonzero(sram_buffer.popleft())
        sram_buffer.append(tensor)
        sram_size_left -= n
        stats_dict[f'miss_{var}'] += 1
        stats_dict[f'hit_{var}'] += n-1

        #! In LoAS, weights and packed spikes have same SRAM behavior.
        stats_dict[f'dram_traffic_{var}b'] += n_tot
        stats_dict[f'sram_traffic_{var}b'] += n_tot
        stats_dict[f'dram_traffic_{var}'] += n
        stats_dict[f'sram_traffic_{var}'] += n
        
        return sram_buffer, sram_size_left, stats_dict

def print_layer_stats(layer_stats):

    #! Set the bitwidth for variables
    bit_dict = {
        'x': 4,
        'w': 8,
        'b': 1,
        'o': 4
    }

    for k,v in layer_stats.items():
        unit = ''
        if 'traffic' in k:
            layer_stats[k] = v*(bit_dict[k[-1]])/(1024*8) #! Convert into KB.
            unit += ' KB'
            if 'dram_traffic' in k:
                layer_stats['total_dram'] += v*(bit_dict[k[-1]])/(1024*8)
            elif 'sram_traffic' in k:
                layer_stats['total_sram'] += v*(bit_dict[k[-1]])/(1024*8)
        elif 'total_dram' in k or 'total_sram' in k:
            unit += ' KB'
        stat_str = f'{k}: {layer_stats[k]}'
        stat_str += unit
        print(stat_str)
    layer_stats['miss_w_rate'] = layer_stats['miss_w']/(layer_stats['hit_w']+layer_stats['miss_w'])
    layer_stats['miss_x_rate'] = layer_stats['miss_x']/(layer_stats['hit_x']+layer_stats['miss_x'])
    print('W-cache miss rate: ', round(layer_stats['miss_w_rate']*100,3))
    print('X-cache miss rate: ', round(layer_stats['miss_x_rate']*100,3))

def loas_sim_layer(inp_mat, w_mat, tiling_dataflow, tile_dict):
    
    x_tiles, w_tiles = _Tile_movement_Custom(inp_mat, w_mat, tiling_dataflow, tile_dict)
    
    layer_stats = {
        'tot_cycs':0,
        'tot_comps':0,
        'sram_traffic_w':0,
        'sram_traffic_x':0,
        'dram_traffic_w':0,
        'dram_traffic_x':0,
        'dram_traffic_o':0,
        'sram_traffic_wb':0,
        'dram_traffic_wb':0,
        'sram_traffic_xb':0,
        'dram_traffic_xb':0,
        'hit_w':0,
        'hit_x':0,
        'miss_w':0,
        'miss_x':0,
        'total_dram':0,
        'total_sram':0
    }


    #! Here the size always means number of weights / inputs, orthogonal to the precision.
    sram_size_w = 128*32
    sram_size_x = 128*1024
    sram_size_w_left = sram_size_w
    sram_size_x_left = sram_size_x
    sram_buffer_w = deque()
    sram_buffer_x = deque()

    #! Previous layer's written back traffic
    layer_stats['dram_traffic_o'] += torch.count_nonzero(inp_mat).item()
    layer_stats['dram_traffic_xb'] += inp_mat.numel()

    for i in tqdm(range(len(x_tiles))):

        sram_buffer_w, sram_size_w_left, layer_stats = _check_dram2sram_traffic(w_tiles[i], sram_buffer_w, sram_size_w_left, layer_stats, 'w')
        sram_buffer_x, sram_size_x_left, layer_stats = _check_dram2sram_traffic(x_tiles[i], sram_buffer_x, sram_size_x_left, layer_stats, 'x')
        
        #! Read the bitmasks of weights and send from SRAM to the PEs
        if not torch.equal(x_tiles[i], x_tiles[i-1]):         
            layer_stats['sram_traffic_xb'] += x_tiles[i].numel()

        x_pe, w_pe, _ = _tile_to_pe_custom(x_tiles[i], w_tiles[i])

        w_nz = torch.count_nonzero(w_tiles[i]).item()
        #! Read the inputs and broadcasted from SRAM to the PEs
        layer_stats['sram_traffic_w'] += w_nz
        layer_stats['sram_traffic_wb'] += w_tiles[i].numel()
        
        latency_lists = []
        for j in (range(len(x_pe))):
            layer_stats, latency = _pe_operate(x_pe[j], w_pe[j], layer_stats)
            latency_lists += [latency]
        layer_stats['tot_cycs'] += max(latency_lists)
    
    print_layer_stats(layer_stats)

    return layer_stats

def test_real_data(path):
    matrices_dict = torch.load(path,map_location=torch.device('cpu'))

    tile_dict = {
        'x': (1,16,128),
        'w': (128,32)
    }
    tiling_dataflow = 'mntk'

    result_dict = {}
    for k,v in matrices_dict.items():
        print('\n')
        print(f"Simulating {k}")
        inp_mat = v['x']
        # print(inp_mat.shape)
        w_mat = v['w']

        result_dict[k] = loas_sim_layer(inp_mat, w_mat, tiling_dataflow, tile_dict)
    
    return result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-simulation")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--loas', type=str, default='strong', help='[normal, strong]')
    parser.add_argument('--spar_lvl', type=str, default='high', help='[high, medium, low]')
    args = parser.parse_args()

    results = test_real_data(f'../matrices_dict/spikeformer_final_matrices_dict_loas_normal.pth')
    torch.save(results, f'./loas-mntk_spikeformer_dict.pth')
    print('Succesfully save the results into the dictionary.')
