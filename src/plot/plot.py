import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, AutoMinorLocator


def plot_vgg16(mode):
    #########################################
    # Data 
    #########################################
    arch = 'vgg16'
    path = './sparten_vgg16_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']

    path = './gospa_vgg16_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    path = './loas-mntk_normal_vgg16_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    path = './loas-mntk_strong_vgg16_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']

    # values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]
    print(arch)
    print('sram: ', [gospa_sram/sparten_sram, gospa_sram/gospa_sram, gospa_sram/loas_sram, gospa_sram/loasft_sram])
    print('dram: ', [gospa_dram/sparten_dram, gospa_dram/gospa_dram, gospa_dram/loas_dram, gospa_dram/loasft_dram])

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('VGG16')
        ax.set_ylabel('Normalized speedup')
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
        plt.savefig('vgg16_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, loas_dram, loasft_dram]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('VGG16')
        ax.set_ylabel('Off-chip memory traffic (KB)')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_sram/1024, gospa_sram/1024, loas_sram/1024, loasft_sram/1024]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('VGG16')
        ax.set_ylabel('On-chip memory traffic (MB)')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_sram.pdf', bbox_inches='tight',pad_inches=0.1)


def plot_alexnet(mode):
    #########################################
    # Data 
    #########################################
    arch = 'alexnet'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']
    
    # values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]
    print(arch)
    # print(values)
    print('sram: ', [gospa_sram/sparten_sram, gospa_sram/gospa_sram, gospa_sram/loas_sram, gospa_sram/loasft_sram])
    print('dram: ', [gospa_dram/sparten_dram, gospa_dram/gospa_dram, gospa_dram/loas_dram, gospa_dram/loasft_dram])

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('Alexnet')
        ax.set_ylabel('Normalized speedup')
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
        plt.savefig(f'{arch}_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, loas_dram, loasft_dram]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('Alexnet')
        ax.set_ylabel('Off-chip memory traffic (KB)')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_sram/1024, gospa_sram/1024, loas_sram/1024, loasft_sram/1024]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('Alexnet')
        # ax.set_ylabel('Speedup')
        ax.set_ylabel('On-chip memory traffic (MB)')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_sram.pdf', bbox_inches='tight',pad_inches=0.1)


def plot_resnet(mode):
    #########################################
    # Data 
    #########################################
    arch = 'resnet19'
    path = f'./sparten_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    sparten_cycles = 0
    sparten_dram = 0
    sparten_sram = 0
    for k,v in dic.items():
        sparten_cycles += v['tot_cycs']
        sparten_dram += v['total_dram']
        sparten_sram += v['total_sram']

    path = f'./gospa_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    gospa_cycles = 0
    gospa_dram = 0
    gospa_sram = 0
    for k,v in dic.items():
        gospa_cycles += v['tot_cycs']
        gospa_dram += v['total_dram']
        gospa_sram += v['total_sram']

    path = f'./loas-mntk_normal_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loas_cycles = 0
    loas_dram = 0
    loas_sram = 0
    for k,v in dic.items():
        loas_cycles += v['tot_cycs']
        loas_dram += v['total_dram']
        loas_sram += v['total_sram']

    path = f'./loas-mntk_strong_{arch}_dict.pth'
    dic = torch.load(path,map_location=torch.device('cpu'))
    loasft_cycles = 0
    loasft_dram = 0
    loasft_sram = 0
    for k,v in dic.items():
        loasft_cycles += v['tot_cycs']
        loasft_dram += v['total_dram']
        loasft_sram += v['total_sram']

    # values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]
    print(arch)
    print('sram: ', [gospa_sram/sparten_sram, gospa_sram/gospa_sram, gospa_sram/loas_sram, gospa_sram/loasft_sram])
    print('dram: ', [gospa_dram/sparten_dram, gospa_dram/gospa_dram, gospa_dram/loas_dram, gospa_dram/loasft_dram])
    # print(values)

    #########################################
    # Plot 
    #########################################
    if mode == 'speedup':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_cycles/sparten_cycles, sparten_cycles/gospa_cycles, sparten_cycles/loas_cycles, sparten_cycles/loasft_cycles]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('ResNet19')
        ax.set_ylabel('Normalized speedup')
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
        plt.savefig(f'{arch}_speedup.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'dram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_dram, gospa_dram, loas_dram, loasft_dram]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('ResNet19')
        ax.set_ylabel('Off-chip memory traffic (KB)')
        # ax.set_ylabel('Speedup')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_dram.pdf', bbox_inches='tight',pad_inches=0.1)
    elif mode == 'sram':
        labels = ['SparTen-S', 'GoSpa-S', 'LoAS', 'LoAS-FT']
        values = [sparten_sram/1024, gospa_sram/1024, loas_sram/1024, loasft_sram/1024]

        # Create bar chart
        fig, ax = plt.subplots(1,1,figsize=(1.2,1.8),dpi=150)
        bars = ax.bar(labels, values, color=['#40679E','#F5DD61','#BC7FCD','#FB9AD1'],linewidth=0.9,zorder=4,edgecolor='black',alpha=0.85)

        # Set title and y-axis label
        ax.set_title('ResNet19')
        # ax.set_ylabel('Speedup')
        ax.set_ylabel('On-chip memory traffic (MB)')
        # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)

        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        # ax.set_yticklabels([])
        plt.xticks(rotation=90)
        plt.savefig(f'{arch}_sram.pdf', bbox_inches='tight',pad_inches=0.1)


if __name__ == '__main__':
    mode = 'sram'
    plot_vgg16(mode)
    plot_alexnet(mode)
    plot_resnet(mode)

