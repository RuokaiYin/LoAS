import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, AutoMinorLocator


def plot_speedup_memtraffic(mode, arch, plot_mode):
    #########################################
    # Data 
    #########################################
    path = f'../simulation_results/raw/sparten/sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']

    path = f'../simulation_results/raw/gospa/gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']
    
    path = f'./gamma_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gamma_cycles = 0
    gamma_dram = 0
    gamma_sram = 0
    for k,v in dic.items():
        gamma_cycles += v['tot_cycs']
        gamma_dram += v['total_dram']
        gamma_sram += v['total_sram']

    path = f'../simulation_results/raw/loas/loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    path = f'../simulation_results/raw/loas/loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']

    # values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]
    # print(arch)
    # print('sram: ', [gospa_sram/sparten_sram, gospa_sram/gospa_sram, gospa_sram/loas_sram, gospa_sram/loasft_sram])
    # print('dram: ', [gospa_dram/sparten_dram, gospa_dram/gospa_dram, gospa_dram/loas_dram, gospa_dram/loasft_dram])

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
        values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/gamma_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#81A263','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        if arch == 'vgg16':
            ax.set_title('VGG16')
        elif arch == 'resnet19':
            ax.set_title('ResNet19')
        elif arch == 'alexnet':
            ax.set_title('Alexnet')

        ax.set_ylabel('Normalized speedup')
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
            plt.savefig(f'../figures/revision/{arch}_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, gamma_dram, loas_dram, loasft_dram]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#81A263','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        if arch == 'vgg16':
            ax.set_title('VGG16')
        elif arch == 'resnet19':
            ax.set_title('ResNet19')
        elif arch == 'alexnet':
            ax.set_title('Alexnet')
        ax.set_ylabel('Off-chip memory traffic (KB)')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        if plot_mode == 'show':
            plt.show()
        else:
            plt.savefig(f'../figures/revision/{arch}_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':
        labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
        values = [sparten_sram/1024, gospa_sram/1024, gamma_sram/1024, loas_sram/1024, loasft_sram/1024]
        print(values)
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#81A263','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        if arch == 'vgg16':
            ax.set_title('VGG16')
            ax.set_ylim(top=100)
        elif arch == 'resnet19':
            ax.set_title('ResNet19')
            ax.set_ylim(top=150)
        elif arch == 'alexnet':
            ax.set_title('Alexnet')
            ax.set_ylim(top=28)
        ax.set_ylabel('On-chip memory traffic (MB)')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        if plot_mode == 'show':
            plt.show()
        else:
            plt.savefig(f'../figures/revision/{arch}_sram.pdf', bbox_inches='tight',pad_inches=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-Plot")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--mode', type=str, default='speedup', help='[speedup, dram, sram]')
    parser.add_argument('--pmode', type=str, default='show', help='[show, save]')
    args = parser.parse_args()
    plot_speedup_memtraffic(args.mode, args.arch, args.pmode)
