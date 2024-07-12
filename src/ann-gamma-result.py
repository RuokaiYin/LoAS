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

    #! In Gamma, the inter-PE dimensions for w is 0, meaning all PE share same w.
    #! In Gamma, the inter-PE dim for x is x.size(1), meaning PEs divide the inputs' rows.
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

    #! Modeling the memory movement
    #! Firstly, for each tile, read the non-zero weight values (Bkn).
    stats_dict['sram_traffic_w'] += torch.count_nonzero(w).item()
    #! And their coordinates (N)
    stats_dict['sram_traffic_w_b'] += torch.count_nonzero(w).item()

    #! Read the input from each rows (Amk)
    stats_dict['sram_traffic_x'] += torch.count_nonzero(x).item()
    #! And the inputs coordinates (M)
    stats_dict['sram_traffic_x_a'] += torch.count_nonzero(x).item()

    #! TODO: model the partial row movement (done)
    #! Calculate the number of non-zero columns of weights. That is where the accumulation needs to happen.
    w_nzcol = torch.count_nonzero(torch.sum(w,dim=0))

    
    #! Important, since we assume there will be a small SRAM buffer in Gamma's design to capture psum row.
    #! So, we can assume, 1/t of the psum rows will be buffered into the SRAM. (1/t can be adjusted by different strength of estimation, for example, 2/t is also possible, but increase this has the tradeoff of increasing SRAM size.)
    #! The other (t-1)/t will be first written into SRAM, than evict back to DRAM.
    #! To more detailedly modeling, we need to have sram traffic times 4 for modeling sram<->dram, but here for estimation, we only use 2 times.
    stats_dict['sram_traffic_o'] += w_nzcol*2 #! 2 times, 1 for reading to merge, 1 for writing

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
        stats_dict[f'dram_traffic_{var}'] += n
        stats_dict[f'sram_traffic_{var}'] += n
        if var == 'w':
            stats_dict[f'dram_traffic_{var}_b'] += n
            stats_dict[f'sram_traffic_{var}_b'] += n
        elif var == 'x':
            stats_dict[f'dram_traffic_{var}_a'] += n #! Already considered Prev layer's written back, in layer_sim.
            stats_dict[f'sram_traffic_{var}_a'] += n
        else:
            raise Exception("SRAM type other than w or x is provided.")
        
    return sram_buffer, sram_size_left, stats_dict

def print_layer_stats(layer_stats):

    # x_bit = 1
    # w_bit = 8
    # o_bit = 8
    # w_csr_bit = 8 #* Denote as b
    # x_csr_bit = 8 #* Denote as a
    #! Set the bitwidth for variables
    bit_dict = {
        'x': 8,
        'w': 8,
        'a': 6, #! a indicates csr for x
        'b': 6, #! b indicates csr for w
        'o': 8
    }
    for k,v in layer_stats.items():
        unit = ''
        if 'traffic' in k:
            layer_stats[k] = v*(bit_dict[k[-1]])/(1024*8) #! First decode the bitwidth for the value, then Convert into KB.
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
    #! TODO: add psum miss rate, but currently its not necessary here.
    print('W-cache miss rate: ', round(layer_stats['miss_w_rate']*100,3))
    print('X-cache miss rate: ', round(layer_stats['miss_x_rate']*100,3))

def gamma_sim_layer(inp_mat, w_mat, tiling_dataflow, tile_dict):
    
    x_tiles, w_tiles = _Tile_movement_Custom(inp_mat, w_mat, tiling_dataflow, tile_dict)
    
    layer_stats = {
        'tot_cycs':0,
        'tot_comps':0,
        'sram_traffic_w':0,
        'sram_traffic_x':0,
        'sram_traffic_o':0,
        'dram_traffic_w':0,
        'dram_traffic_x':0,
        'dram_traffic_o':0,
        'sram_traffic_x_a':0,
        'sram_traffic_w_b':0,
        'dram_traffic_x_a':0,
        'dram_traffic_w_b':0,
        'hit_w':0,
        'hit_x':0,
        'miss_w':0,
        'miss_x':0,
        'total_dram':0,
        'total_sram':0
    }

    #! Here the size always means number of weights / inputs, orthogonal to the precision.
    #! Gamma has a way more larger SRAM (3MB) than our design choice due to its operation on neumeric sparse data usually has 64bits precision.
    #! Here we scale it down, Gamma has in total 3MB SRAM cache. We use 8 bit quantities, so 3MB/8 -> 384KB. In our design, we use 256KB (double buffered).
    #! So in pratical, we have roughly 132 KB of data space for operating in LoAS design. Here we should scale it.
    #! The gamma has 32 PEs, each operate on one input row, so 32, R = 64 in gamma, thats 32*64, that's 2KB input. We scale along the k dimension, 32*(64*4). More scale (32)*(64*4)
    #! We set weight to be 64*256, to align with GoSPA, thats 16KB. To scale with k in x, we should have 256*512.
    #! In total now is 136 KB, assuming some of the space are left for psum rows. But in reality, this already has larger SRAM size than LoAS. So we give a slightly of free psum SRAM to Gamma.
    sram_size_w = 256*512
    sram_size_x = 32*128
    sram_size_w_left = sram_size_w
    sram_size_x_left = sram_size_x
    sram_buffer_w = deque()
    sram_buffer_x = deque()
    #! Prev layer's csr dram writing for writing back the final output.
    layer_stats[f'dram_traffic_x_a'] += torch.count_nonzero(inp_mat).item()
    layer_stats['dram_traffic_o'] += torch.count_nonzero(inp_mat).item()
    #! Todo, depends on simulation results, determine whether we should add dram_traffic_o
    
    for i in tqdm(range(len(x_tiles))):

        sram_buffer_w, sram_size_w_left, layer_stats = _check_dram2sram_traffic(w_tiles[i], sram_buffer_w, sram_size_w_left, layer_stats, 'w')
        sram_buffer_x, sram_size_x_left, layer_stats = _check_dram2sram_traffic(x_tiles[i], sram_buffer_x, sram_size_x_left, layer_stats, 'x')
        
        x_pe, w_pe, _ = _tile_to_pe_custom(x_tiles[i], w_tiles[i])
        
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
        'x': (1,16,64),
        'w': (64,64)
    }
    tiling_dataflow = 'mktn'

    result_dict = {}
    #! for debugging
    dram = 0
    sram = 0
    for k,v in matrices_dict.items():
        print('\n')
        print(f"Simulating {k}")
        inp_mat = v['x']
        w_mat = v['w']

        result_dict[k] = gamma_sim_layer(inp_mat, w_mat, tiling_dataflow, tile_dict)
        dram += result_dict[k]['total_dram']
        sram += result_dict[k]['total_sram']
    print(f'Final DRAM: {dram}')
    print(f'Final SRAM: {sram}')
    return result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ANN-Gamma-simulation")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    args = parser.parse_args()

    results = test_real_data(f'../matrices_dict/{args.arch}_ann_final_matrices_dict.pth')
    torch.save(results, f'./gamma_ann_{args.arch}_dict.pth')
    print('Succesfully save the results into the dictionary.')
