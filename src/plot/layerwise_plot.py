import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, AutoMinorLocator


def plot_layerwise(arch, mode):
    #########################################
    # Data 
    #########################################
    # arch = 'alexnet'
    if arch == 'alexnet':
        lay = 'Layer_3'
    elif arch == 'vgg16':
        lay = 'Layer_7'
    elif arch == 'resnet19':
        lay = 'Layer_15'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_dram = 0
    sparten_sram = 0

    sparten_dram_w = 0
    sparten_dram_x = 0
    sparten_dram_o = 0
    sparten_dram_others = 0
    
    sparten_sram_w = 0
    sparten_sram_x = 0
    sparten_sram_o = 0
    sparten_sram_others = 0
    sparten_cache_miss_x = 0
    sparten_cache_miss_w = 0
    for k,v in dic.items():
        if k == lay:
            sparten_dram += v['total_dram']
            sparten_sram += v['total_sram']

            sparten_dram_w += v['dram_traffic_w']
            sparten_dram_x += v['dram_traffic_x']
            sparten_dram_o += v['dram_traffic_o']
            sparten_dram_others += v['dram_traffic_wb']

            sparten_sram_w += v['sram_traffic_w']
            sparten_sram_x += v['sram_traffic_x']
            sparten_sram_others += v['sram_traffic_wb']

            sparten_cache_miss_x += v['miss_x']/(v['hit_x']+v['miss_x'])*100
            sparten_cache_miss_w += v['miss_w']/(v['hit_w']+v['miss_w'])*100

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_dram = 0
    gospa_dram_w = 0
    gospa_dram_x = 0
    gospa_dram_o = 0
    gospa_dram_others = 0
    
    gospa_sram = 0
    gospa_sram_w = 0
    gospa_sram_x = 0
    gospa_sram_o = 0
    gospa_sram_others = 0
    gospa_cache_miss_x = 0
    gospa_cache_miss_w = 0
    for k,v in dic.items():
        if k == lay:
            gospa_cycles += v['tot_cycs']
            gospa_dram += v['total_dram']
            gospa_sram += v['total_sram']
            gospa_dram_w += v['dram_traffic_w']
            gospa_dram_x += v['dram_traffic_x']
            gospa_dram_o += v['dram_traffic_o']
            gospa_dram_others += v['dram_traffic_wsp']
            gospa_dram_others += v['dram_traffic_csr']
                    
            gospa_sram_w += v['sram_traffic_w']
            gospa_sram_x += v['sram_traffic_x']
            gospa_sram_others += v['sram_traffic_wsp']
            gospa_sram_others += v['sram_traffic_csr']

            gospa_cache_miss_x += v['miss_x']/(v['hit_x']+v['miss_x'])*100
            gospa_cache_miss_w += v['miss_w']/(v['hit_w']+v['miss_w'])*100

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        if k == lay:
            loas_cycles += v['tot_cycs']
            loas_dram += v['total_dram']
            loas_sram += v['total_sram']

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_dram = 0
    loasft_dram_w = 0
    loasft_dram_x = 0
    loasft_dram_o = 0
    loasft_dram_others = 0

    loasft_sram = 0
    loasft_sram_w = 0
    loasft_sram_x = 0
    loasft_sram_o = 0
    loasft_sram_others = 0
    loasft_cache_miss_x = 0
    loasft_cache_miss_w = 0
    for k,v in dic.items():
        if k == lay:
            loasft_cycles += v['tot_cycs']
            loasft_dram += v['total_dram']
            loasft_dram_w += v['dram_traffic_w']
            loasft_dram_x += v['dram_traffic_x']
            loasft_dram_o += v['dram_traffic_o']
            loasft_dram_others += v['dram_traffic_wb']
            loasft_dram_others += v['dram_traffic_xb']
            
            loasft_sram += v['total_sram']
            loasft_sram_w += v['sram_traffic_w']
            loasft_sram_x += v['sram_traffic_x']
            loasft_sram_others += v['sram_traffic_wb']
            loasft_sram_others += v['sram_traffic_xb']

            loasft_cache_miss_x += v['miss_x']/(v['hit_x']+v['miss_x'])*100
            loasft_cache_miss_w += v['miss_w']/(v['hit_w']+v['miss_w'])*100
    
    # values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]
    print(arch)
    print('dram ot: ', [sparten_dram_others/sparten_dram_others, gospa_dram_others/sparten_dram_others, loasft_dram_others/sparten_dram_others])
    # print('x-cache:', [sparten_cache_miss_x,gospa_cache_miss_x,loasft_cache_miss_x])
    # print('w-cache:', [sparten_cache_miss_w,gospa_cache_miss_w,loasft_cache_miss_w])
    # print('sram: ', [gospa_sram/sparten_sram, gospa_sram/gospa_sram, gospa_sram/loas_sram, gospa_sram/loasft_sram])
    # print('dram: ', [gospa_dram/sparten_dram, gospa_dram/gospa_dram, gospa_dram/loas_dram, gospa_dram/loasft_dram])

    #########################################
    # Plot 
    #########################################
    labels = ['SparTen-S', 'GoSpa-S', 'LoAS']
    if mode == 'dram':
        values_x = [sparten_dram_x/loasft_dram, gospa_dram_x/loasft_dram, loasft_dram_x/loasft_dram]
        values_o = [sparten_dram_o/loasft_dram, gospa_dram_o/loasft_dram, loasft_dram_o/loasft_dram]
        values_ot = [sparten_dram_others/loasft_dram, gospa_dram_others/loasft_dram, loasft_dram_others/loasft_dram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.5,3),dpi=150)
        bars = ax.bar(labels, values_w, color='#F4CE14',linewidth=0.9,edgecolor='black',alpha=0.85)
        bars = ax.bar(labels, values_x, bottom =values_w, color='#A3B763',linewidth=0.9,edgecolor='black',alpha=0.85)
        bars = ax.bar(labels, values_o, bottom =np.array(values_w)+np.array(values_x), color='#557C55',linewidth=0.9,edgecolor='black',alpha=0.85)
        bars = ax.bar(labels, values_ot, bottom =np.array(values_w)+np.array(values_x)+np.array(values_o), color='#FD841F',linewidth=0.9,edgecolor='black',alpha=0.85)
    elif mode == 'sram':
        values_w = [sparten_sram_w/loasft_sram, gospa_sram_w/loasft_sram, loasft_sram_w/loasft_sram]
        values_x = [sparten_sram_x/loasft_sram, gospa_sram_x/loasft_sram, loasft_sram_x/loasft_sram]
        values_ot = [sparten_sram_others/loasft_sram, gospa_sram_others/loasft_sram, loasft_sram_others/loasft_sram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.5,3),dpi=150)
        bars = ax.bar(labels, values_w, color='#F4CE14',linewidth=0.9,edgecolor='black',alpha=0.85)
        bars = ax.bar(labels, values_x, bottom =values_w, color='#A3B763',linewidth=0.9,edgecolor='black',alpha=0.85)
        bars = ax.bar(labels, values_ot, bottom =np.array(values_w)+np.array(values_x), color='#FD841F',linewidth=0.9,edgecolor='black',alpha=0.85)
    elif mode == 'miss':
        miss_x = [sparten_cache_miss_x/loasft_cache_miss_x,gospa_cache_miss_x/loasft_cache_miss_x,loasft_cache_miss_x/loasft_cache_miss_x]
        # miss_w = [sparten_cache_miss_w,gospa_cache_miss_w,loasft_cache_miss_w]
        width = 0.2
        x= np.array([1,2,3])
        fig, ax = plt.subplots(1,1,figsize=(1.5,3),dpi=150)
        bars = ax.bar(labels, miss_x, color='#B99470',linewidth=0.9,edgecolor='black',alpha=0.85, width=0.9)
        ax.set_ylim(0,2)
        ax.set_yticklabels([])
        # bars = ax.bar(x-width, miss_w, color='#A3B763',linewidth=0.9,edgecolor='black',alpha=0.85, width=0.4)

    # Set title and y-axis label
    if arch == 'alexnet':
        ax.set_title('Alexnet-L3')
    elif arch == 'vgg16':
        ax.set_title('VGG16-L7')
    elif arch == 'resnet19':
        ax.set_title('ResNet19-L19')
    
    if mode == 'dram':
        ax.set_ylabel('Normalized off-chip traffic')
    elif mode == 'sram':
        ax.set_ylabel('Normalized on-chip traffic')
    elif mode == 'cache':
        ax.set_ylabel('Normalized cache miss rate')
    # ax.set_ylabel('Speedup')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_yticklabels([])
    # ax.set_ylim(1,20)
    # ax.set_xticklabels(['SparTen-S', 'GoSpa-S', 'LoAS'])
    plt.xticks(rotation=90)


    # Display the figure
    # plt.show()
    # plt.savefig(f'{arch}_layerwise_{mode}.pdf', bbox_inches='tight',pad_inches=0.1)



if __name__ == '__main__':
    mode = 'miss'
    plot_layerwise('resnet19',mode)
    plot_layerwise('alexnet',mode)
    plot_layerwise('vgg16',mode)
