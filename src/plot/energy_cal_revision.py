# Total dpower of 2.3214079694375 mw
# Total lpower of 0.4951561001875001 mW
# Total energy of 5.8035199235937505 nJ
# SparTen data
# Total area of 0.036605763686499995 mm^3
# Total dpower of 2.4597227960000003 mw
# Total lpower of 0.49841297987500005 mW
# Total energy of 6.14930699 nJ
# GoSPA data
# Total area of 0.386209676437 mm^3
# Total dpower of 31.518842075000002 mw
# Total lpower of 7.711707425 mW
# Total energy of 1.3409703593750002 nJ

# Gamma data
# Total area of 0.019422482042 mm^3
# Total dpower of 1.9950422250000002 mW
# Total lpower of 0.32129172500000003 mW
# Total energy of 4.9876055625 nJ


import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
import numpy as np
import argparse

loas_comp = 5.8
sparten_comp = 6.15
gospa_comp = 1.34 * 1.3 #! 30% IDGEN overhead
sparten_ann_comp = 7.949
ptb_comp = 0.148*16*4
gamma_comp = 4.98*(6/8)

# left: comp, right: sram
loas_lp = 0.5 + 124.48
sparten_lp = 0.5 + 124.48
gospa_lp = 31 + 148
sparten_ann_lp = 0.92 + 130 #! slightly increase cache size for larger input map
ptb_lp = 0.016*16*4 + 78 #! 162 KB glb
gamma_lp = 0.32 + 148



c_time = 1/800
#! This is the one that used in snn and snn comparison, both loas and sparten use this dram energy
dram_e = 2.9
#! This is the one that used in ann and snn comparison, both loas and sparten use this dram energy
ann_dram = 9.6
# ann_dram = 2.9
sram_e = 0.0681

def plot_vgg16():
    #########################################
    # Data 
    #########################################
    arch = 'vgg16'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_comps = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_comps += v['tot_comps']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']
    
    sparten_e = sparten_cycles*c_time*sparten_lp + dram_e*sparten_dram*1024 + sram_e*sparten_sram*1024 + sparten_comp*sparten_comps

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_comps = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_comps += v['tot_comps']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    gospa_e = gospa_cycles*c_time*gospa_lp + dram_e*gospa_dram*1024 + sram_e*gospa_sram*1024 + gospa_comp*gospa_comps

    path = f'./gamma_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_comps = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_comps += v['tot_comps']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    gospa_e = gospa_cycles*c_time*gospa_lp + dram_e*gospa_dram*1024 + sram_e*gospa_sram*1024 + gospa_comp*gospa_comps

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_comps = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_comps += v['tot_comps']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    loas_e = loas_cycles*c_time*loas_lp + dram_e*loas_dram*1024 + sram_e*loas_sram*1024 + loas_comp*loas_comps

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + dram_e*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps

    print(arch)
    print([sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e])
    print([sparten_cycles*c_time*sparten_lp/sparten_e, dram_e*sparten_dram*1024/sparten_e, sram_e*sparten_sram*1024/sparten_e, sparten_comp*sparten_comps/sparten_e])
    print([gospa_cycles*c_time*gospa_lp/gospa_e, dram_e*gospa_dram*1024/gospa_e, sram_e*gospa_sram*1024/gospa_e, gospa_comp*gospa_comps/gospa_e])
    print([loasft_cycles*c_time*loas_lp/loasft_e, dram_e*loasft_dram*1024/loasft_e, sram_e*loasft_sram*1024/loasft_e, loas_comp*loasft_comps/loasft_e])

    #########################################
    # Plot 
    #########################################

    labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
    values = [sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e]

    # Create bar chart
    fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
    bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

    # Set title and y-axis label
    ax.set_title('VGG16')
    ax.set_ylabel('Normalized energy efficiency')
    # ax.set_ylabel('Speedup')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_yticklabels([])
    plt.xticks(rotation=90)

    # Display the figure
    plt.show()
    # plt.savefig('vgg16_energy.pdf', bbox_inches='tight',pad_inches=0.1)

def plot_alexnet():
    #########################################
    # Data 
    #########################################
    arch = 'alexnet'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_comps = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_comps += v['tot_comps']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']
    
    sparten_e = sparten_cycles*c_time*sparten_lp + dram_e*sparten_dram*1024 + sram_e*sparten_sram*1024 + sparten_comp*sparten_comps
    

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_comps = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_comps += v['tot_comps']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    gospa_e = gospa_cycles*c_time*gospa_lp + dram_e*gospa_dram*1024 + sram_e*gospa_sram*1024 + gospa_comp*gospa_comps
    # print([gospa_cycles*c_time*gospa_lp/gospa_e, dram_e*gospa_dram*1024/gospa_e, sram_e*gospa_sram*1024/gospa_e, gospa_comp*gospa_comps/gospa_e])

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_comps = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_comps += v['tot_comps']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    loas_e = loas_cycles*c_time*loas_lp + dram_e*loas_dram*1024 + sram_e*loas_sram*1024 + loas_comp*loas_comps

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + dram_e*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps
    

    print(arch)
    print([sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e])
    print([sparten_cycles*c_time*sparten_lp/sparten_e, dram_e*sparten_dram*1024/sparten_e, sram_e*sparten_sram*1024/sparten_e, sparten_comp*sparten_comps/sparten_e])
    print([gospa_cycles*c_time*gospa_lp/gospa_e, dram_e*gospa_dram*1024/gospa_e, sram_e*gospa_sram*1024/gospa_e, gospa_comp*gospa_comps/gospa_e])
    print([loasft_cycles*c_time*loas_lp/loasft_e, dram_e*loasft_dram*1024/loasft_e, sram_e*loasft_sram*1024/loasft_e, loas_comp*loasft_comps/loasft_e])
    

    #########################################
    # Plot 
    #########################################
    labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
    values = [sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e]

    # Create bar chart
    fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
    bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

    # Set title and y-axis label
    ax.set_title('Alexnet')
    ax.set_ylabel('Normalized energy efficiency')
    # ax.set_ylabel('Speedup')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_yticklabels([])
    plt.xticks(rotation=90)

    # Display the figure
    # plt.show()
    # plt.savefig(f'{arch}_energy.pdf', bbox_inches='tight',pad_inches=0.1)

def plot_resnet():
    #########################################
    # Data 
    #########################################
    arch = 'resnet19'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_comps = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_comps += v['tot_comps']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']
    
    sparten_e = sparten_cycles*c_time*sparten_lp + dram_e*sparten_dram*1024 + sram_e*sparten_sram*1024 + sparten_comp*sparten_comps

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_comps = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_comps += v['tot_comps']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    gospa_e = gospa_cycles*c_time*gospa_lp + dram_e*gospa_dram*1024 + sram_e*gospa_sram*1024 + gospa_comp*gospa_comps

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_comps = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_comps += v['tot_comps']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    loas_e = loas_cycles*c_time*loas_lp + dram_e*loas_dram*1024 + sram_e*loas_sram*1024 + loas_comp*loas_comps

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + dram_e*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps

    print(arch)
    print([sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e])
    print([sparten_cycles*c_time*sparten_lp/sparten_e, dram_e*sparten_dram*1024/sparten_e, sram_e*sparten_sram*1024/sparten_e, sparten_comp*sparten_comps/sparten_e])
    print([gospa_cycles*c_time*gospa_lp/gospa_e, dram_e*gospa_dram*1024/gospa_e, sram_e*gospa_sram*1024/gospa_e, gospa_comp*gospa_comps/gospa_e])
    print([loasft_cycles*c_time*loas_lp/loasft_e, dram_e*loasft_dram*1024/loasft_e, sram_e*loasft_sram*1024/loasft_e, loas_comp*loasft_comps/loasft_e])

    #########################################
    # Plot 
    #########################################
    labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
    values = [sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/loas_e, sparten_e/loasft_e]

    # Create bar chart
    fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
    bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

    # Set title and y-axis label
    ax.set_title('ResNet19')
    ax.set_ylabel('Normalized energy efficiency')
    # ax.set_ylabel('Speedup')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_yticklabels([])
    plt.xticks(rotation=90)

    # Display the figure
    # plt.show()
    plt.savefig(f'{arch}_energy.pdf', bbox_inches='tight',pad_inches=0.1)


def plot_snn_energy(arch, plot_mode):
    #########################################
    # Data 
    #########################################
    path = f'../simulation_results/raw/sparten/sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_comps = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_comps += v['tot_comps']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']
    
    sparten_e = sparten_cycles*c_time*sparten_lp + dram_e*sparten_dram*1024 + sram_e*sparten_sram*1024 + sparten_comp*sparten_comps

    path = f'../simulation_results/raw/gospa/gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_comps = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_comps += v['tot_comps']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    gospa_e = gospa_cycles*c_time*gospa_lp + dram_e*gospa_dram*1024 + sram_e*gospa_sram*1024 + gospa_comp*gospa_comps

    path = f'./gamma_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gamma_cycles = 0
    gamma_comps = 0
    gamma_dram = 0
    gamma_sram = 0
    for k,v in dic.items():
        gamma_cycles += v['tot_cycs']
        gamma_comps += v['tot_comps']
        gamma_dram += v['total_dram']
        gamma_sram += v['total_sram']

    gamma_e = gamma_cycles*c_time*gamma_lp + dram_e*gamma_dram*1024 + sram_e*gamma_sram*1024 + gamma_comp*gamma_comps

    path = f'../simulation_results/raw/loas/loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_comps = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_comps += v['tot_comps']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    loas_e = loas_cycles*c_time*loas_lp + dram_e*loas_dram*1024 + sram_e*loas_sram*1024 + loas_comp*loas_comps

    path = f'../simulation_results/raw/loas/loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + dram_e*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps

    print(arch)
    print([sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/gamma_e, sparten_e/loas_e, sparten_e/loasft_e])
    print([sparten_cycles*c_time*sparten_lp/sparten_e, dram_e*sparten_dram*1024/sparten_e, sram_e*sparten_sram*1024/sparten_e, sparten_comp*sparten_comps/sparten_e])
    print([gospa_cycles*c_time*gospa_lp/gospa_e, dram_e*gospa_dram*1024/gospa_e, sram_e*gospa_sram*1024/gospa_e, gospa_comp*gospa_comps/gospa_e])
    print([gamma_cycles*c_time*gamma_lp/gamma_e, dram_e*gamma_dram*1024/gamma_e, sram_e*gamma_sram*1024/gamma_e, gamma_comp*gamma_comps/gamma_e])
    print([loasft_cycles*c_time*loas_lp/loasft_e, dram_e*loasft_dram*1024/loasft_e, sram_e*loasft_sram*1024/loasft_e, loas_comp*loasft_comps/loasft_e])

    #########################################
    # Plot 
    #########################################

    labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
    values = [sparten_e/sparten_e, sparten_e/gospa_e, sparten_e/gamma_e, sparten_e/loas_e, sparten_e/loasft_e]

    # Create bar chart
    fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
    bars = ax.bar(labels, values, color=['#40679E','#F5DD61', '#81A263', '#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

    # Set title and y-axis label
    if arch == 'vgg16':
        ax.set_title('VGG16')
    elif arch == 'resnet19':
        ax.set_title('ResNet19')
    elif arch == 'alexnet':
        ax.set_title('Alexnet')
    ax.set_ylabel('Normalized energy efficiency')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_yticklabels([])
    plt.xticks(rotation=90)

    # Display the figure
    if plot_mode == 'show':
        plt.show()
    else:
        plt.savefig(f'../figures/revision/{arch}_energy.pdf', bbox_inches='tight',pad_inches=0.1)

def plot_ann(arch, mode):

    #########################################
    # Data 
    #########################################

    path = f'./sparten_ann_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_ann_cycles = 0
    sparten_ann_comps = 0
    sparten_ann_dram = 0
    sparten_ann_sram = 0
    for k,v in dic.items():
        sparten_ann_cycles += v['tot_cycs']
        sparten_ann_comps += v['tot_comps']
        sparten_ann_dram += v['total_dram']
        sparten_ann_sram += v['total_sram']
    
    sparten_ann_e = sparten_ann_cycles*c_time*sparten_ann_lp + ann_dram*sparten_ann_dram*1024 + sram_e*sparten_ann_sram*1024 + sparten_ann_comp*sparten_ann_comps
    print('ann breakup:',[(ann_dram*sparten_ann_dram*1024 + sram_e*sparten_ann_sram*1024)/sparten_ann_e])


    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + ann_dram*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps
    print('snn breakup:',[(ann_dram*loasft_dram*1024 + sram_e*loasft_sram*1024)/loasft_e])



    #########################################
    # Plot 
    #########################################
    labels = ['SparTen','LoAS']
    if mode == 'energy':
        values = [sparten_ann_e/sparten_ann_e, sparten_ann_e/loasft_e]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,2),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('Energy Efficiency')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        # plt.show()
        plt.savefig(f'{arch}_ann_energy.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':

        values = [sparten_ann_dram/sparten_ann_dram, loasft_dram/sparten_ann_dram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,2),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('DRAM Traffic')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        # plt.show()
        plt.savefig(f'{arch}_ann_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':

        values = [sparten_ann_sram/sparten_ann_sram, loasft_sram/sparten_ann_sram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,2),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('SRAM Traffic')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        # plt.show()
        plt.savefig(f'{arch}_ann_sram.pdf', bbox_inches='tight',pad_inches=0.1)

    elif mode == 'compute':

        values = [(sparten_ann_comp*sparten_ann_comps)/(sparten_ann_comp*sparten_ann_comps), (loas_comp*loasft_comps)/(sparten_ann_comp*sparten_ann_comps)]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,2.5),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('SRAM Traffic')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        plt.show()
        # plt.savefig(f'{arch}_ann_sram.pdf', bbox_inches='tight',pad_inches=0.1)


def plot_ptb(arch, mode):

    #########################################
    # Data 
    #########################################

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_comps = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_comps += v['tot_comps']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    loasft_e = loasft_cycles*c_time*loas_lp + dram_e*loasft_dram*1024 + sram_e*loasft_sram*1024 + loas_comp*loasft_comps
    print('snn breakup:',[(ann_dram*loasft_dram*1024 + sram_e*loasft_sram*1024)/loasft_e])

    ptb_cycles_l = [1210367,2390015,1195007,2374655,2374655,1187327,2366975,2366975,2366975,2366975,2366975]
    ptb_sram_x_l = [18874368,37748736,18874368,37748736,37748736,18874368,37748736,37748736,9437184,9437184,9437184]
    ptb_sram_w_l = [1179648,2359296,1179648,2359296,2359296,1179648,2359296,2359296,2359296,2359296,2359296]
    ptb_sram_o_l = [67584,67584,33792,33792,33792,16896,16896,16896,10752,10752,10752]
    ptb_dram_x_l = [20736,6183048,12800,25600,25600,9216,18432,18432,8192,8192,8192]
    ptb_dram_w_l = [74784,152976,307056,641712,641712,1179648,2359296,2359296,2359296,2359296,2359296]
    ptb_dram_o_l = [32768,32768,16384,16384,16384,8192,8192,8192,2048,2048,2048]

    ptb_cycles = 0
    ptb_sram_x = 0
    ptb_sram_w = 0
    ptb_sram_o = 0
    ptb_dram_x = 0
    ptb_dram_w = 0
    ptb_dram_o = 0
    for i in ptb_cycles_l:
        ptb_cycles += i
    for i in ptb_sram_x_l:
        ptb_sram_x += i
    for i in ptb_sram_w_l:
        ptb_sram_w += i
    for i in ptb_sram_o_l:
        ptb_sram_o += i
    for i in ptb_dram_x_l:
        ptb_dram_x += i
    for i in ptb_dram_w_l:
        ptb_dram_w += i
    for i in ptb_dram_o_l:
        ptb_dram_o += i

    ptb_e = (ptb_dram_o*4+ptb_dram_w*8+ptb_dram_x*4)/8*dram_e + (ptb_sram_o*4+ptb_sram_w*8+ptb_sram_x*4)/8*sram_e + ptb_cycles*c_time*ptb_lp + ptb_cycles*ptb_comp
    ptb_dram = (ptb_dram_o*4+ptb_dram_w*8+ptb_dram_x*4)/(8*1024)
    ptb_sram = (ptb_sram_o*4+ptb_sram_w*8+ptb_sram_x*4)/(8*1024)
    print(ptb_e/ptb_e, ptb_e/loasft_e)
    print(ptb_dram/ptb_dram, loasft_dram/ptb_dram)
    print(ptb_sram/ptb_sram, loasft_sram/ptb_sram)
    #########################################
    # Plot 
    #########################################
    labels = ['PTB','LoAS']
    if mode == 'energy':
        values = [ptb_e/ptb_e, ptb_e/loasft_e]
        print(values)

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,1.5),dpi=150)
        bars = ax.bar(labels, values, color=['#4793AF','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('Energy')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)
        
        # Display the figure
        
        # plt.show()
        plt.savefig(f'{arch}_ptb_energy.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':

        values = [ptb_dram/ptb_dram, loasft_dram/ptb_dram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,1.5),dpi=150)
        bars = ax.bar(labels, values, color=['#4793AF','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('DRAM')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        # plt.show()
        # plt.savefig(f'{arch}_ptb_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':

        values = [ptb_sram/ptb_sram, loasft_sram/ptb_sram]
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,1.5),dpi=150)
        bars = ax.bar(labels, values, color=['#4793AF','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('SRAM')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        # plt.show()
        # plt.savefig(f'{arch}_ptb_sram.pdf', bbox_inches='tight',pad_inches=0.1)

    elif mode == 'cycles':

        values = [ptb_cycles/ptb_cycles, ptb_cycles/loasft_cycles]
        print(values)
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1,1.5),dpi=150)
        bars = ax.bar(labels, values, color=['#4793AF','#FB9AD1'],linewidth=0.9,edgecolor='black',alpha=0.75)

        # Set title and y-axis label
        ax.set_title('Speedup')
        # ax.set_ylabel('Normalized energy efficiency')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        plt.yscale('log')
        # plt.show()
        plt.savefig(f'{arch}_ptb_speed.pdf', bbox_inches='tight',pad_inches=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-Plot")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--mode', type=str, default='snn', help='[snn, ann, ptb]')
    parser.add_argument('--pmode', type=str, default='show', help='[show, save]')
    args = parser.parse_args()
    if args.mode == 'snn':
        plot_snn_energy(args.arch, args.pmode)

