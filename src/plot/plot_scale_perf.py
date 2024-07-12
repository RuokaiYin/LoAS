import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, AutoMinorLocator


def plot_speedup_memtraffic(mode, plot_mode):
    #########################################
    # Data 
    #########################################
    arch = 'vgg16'
    path = f'./loas-mntk_strong_{arch}_high_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    high_cycles = 0
    high_dram = 0
    high_sram = 0
    for k,v in dic.items():
        high_cycles += v['tot_cycs']
        high_dram += v['total_dram']
        high_sram += v['total_sram']

    path = f'./loas-mntk_strong_{arch}_medium_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    medium_cycles = 0
    medium_dram = 0
    medium_sram = 0
    for k,v in dic.items():
        medium_cycles += v['tot_cycs']
        medium_dram += v['total_dram']
        medium_sram += v['total_sram']


    path = f'./loas-mntk_strong_{arch}_low_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    low_cycles = 0
    low_dram = 0
    low_sram = 0
    for k,v in dic.items():
        low_cycles += v['tot_cycs']
        low_dram += v['total_dram']
        low_sram += v['total_sram']

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['High', 'Medium', 'Low']
        values = [high_cycles/high_cycles, high_cycles/medium_cycles, high_cycles/low_cycles]
        print(values)
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#FB9AD1', '#FB9AD1', '#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.8)

        # Set title and y-axis label
        ax.set_title('B Sparsity')

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
            plt.savefig(f'../figures/revision/{arch}_Bsparscale_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, gamma_dram, loas_dram, loasft_dram]
        print(gamma_dram/loasft_dram)
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
        print(gamma_sram/loasft_sram)
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


def plot_speedup_memtraffic_T(mode, plot_mode):
    #########################################
    # Data 
    #########################################
    arch = 'vgg16'
    path = f'./loas-mntk_strong_{arch}_high_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    high_cycles = 0
    high_dram = 0
    high_sram = 0
    for k,v in dic.items():
        high_cycles += v['tot_cycs']
        high_dram += v['total_dram']
        high_sram += v['total_sram']

    path = f'./loas-mntk_strong_{arch}_high_T6_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    medium_cycles = 0
    medium_dram = 0
    medium_sram = 0
    for k,v in dic.items():
        medium_cycles += v['tot_cycs']
        medium_dram += v['total_dram']
        medium_sram += v['total_sram']


    path = f'./loas-mntk_strong_{arch}_high_T8_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    low_cycles = 0
    low_dram = 0
    low_sram = 0
    for k,v in dic.items():
        low_cycles += v['tot_cycs']
        low_dram += v['total_dram']
        low_sram += v['total_sram']

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['T4', 'T6', 'T8']
        values = [high_cycles/high_cycles, high_cycles/medium_cycles, high_cycles/low_cycles]
        print(values)
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#FB9AD1', '#FB9AD1', '#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.8)

        # Set title and y-axis label
        ax.set_title('Timesteps')

        ax.set_ylabel('Normalized speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        if plot_mode == 'show':
            plt.show()
        else:
            plt.savefig(f'../figures/revision/{arch}_timescale_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'Gamma-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, gamma_dram, loas_dram, loasft_dram]
        print(gamma_dram/loasft_dram)
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
        print(gamma_sram/loasft_sram)
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


def plot_speedup_trans(mode, plot_mode):
    #########################################
    # Data 
    #########################################
    arch = 'vgg16'
    lay = 'Layer_7'
    path = f'./loas-mntk_strong_{arch}_high_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    high_cycles = 0
    high_dram = 0
    high_sram = 0
    
    for k,v in dic.items():
        if k == lay:
            high_cycles += v['tot_cycs']
            high_dram += v['total_dram']
            high_sram += v['total_sram']

    path = f'./loas-mntk_spikeformer_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    medium_cycles = 0
    medium_dram = 0
    medium_sram = 0
    for k,v in dic.items():
        medium_cycles += v['tot_cycs']
        medium_dram += v['total_dram']
        medium_sram += v['total_sram']

    vgg_w = 2304*512*16
    trans_w = 3072*3072*784

    high_cycles = high_cycles/vgg_w
    medium_cycles = medium_cycles/trans_w
    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['V-L8', 'T-HFF']
        values = [high_cycles/high_cycles, medium_cycles/high_cycles]
        print(values)
        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#FB9AD1', '#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.8)

        # Set title and y-axis label
        ax.set_title('Layer Size')

        ax.set_ylabel('Normalized Cycles/MAC')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        # plt.xticks(rotation=90)

        # Display the figure
        if plot_mode == 'show':
            plt.show()
        else:
            plt.savefig(f'../figures/revision/transformer_speedup.pdf', bbox_inches='tight',pad_inches=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LoAS-Plot")
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--mode', type=str, default='speedup', help='[speedup, dram, sram]')
    parser.add_argument('--pmode', type=str, default='show', help='[show, save]')
    args = parser.parse_args()
    # plot_speedup_memtraffic(args.mode, args.pmode)
    # plot_speedup_memtraffic_T(args.mode, args.pmode)
    plot_speedup_trans(args.mode, args.pmode)
