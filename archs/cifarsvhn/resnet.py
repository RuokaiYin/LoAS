import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

tau_global = 1./(1. - 0.25)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.total_timestep = 4

        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    #* The implementation of running the LIF in parallel during the forward pass.
    def _parallel_lif(self,lif_module, p_in,  block=0):
        parallel_spikes = 0
        mask = 0
        b = int(p_in.shape[0]/self.total_timestep)
        for t in range(self.total_timestep):
            s_t = lif_module(p_in[t*b:(t+1)*b])
            mask += s_t
            if t == 0:
                parallel_spikes = s_t
            else:
                parallel_spikes = torch.cat((parallel_spikes,s_t))
        
        #! Define your masks here, we provide the one to mask out 1 spike or 2 spikes.
        mask_1 = (mask != 1.0).float().repeat(self.total_timestep, 1, 1, 1)
        mask_2 = (mask != 2.0).float().repeat(self.total_timestep, 1, 1, 1)
        if block == 0:
            return parallel_spikes
        elif block == 1:
            return parallel_spikes*mask_1
        elif block == 2:
            return parallel_spikes*mask_1*mask_2
        else:
            raise Exception("Sorry, the configured mask is not supported!!")

    def forward(self, x):
        x,mask = x
        out = self.bn1(self.conv1(x))
        p_spikes = self._parallel_lif(self.lif1,out,mask)
        out = self.bn2(self.conv2(p_spikes))
        out += self.shortcut(x)
        p_spikes = self._parallel_lif(self.lif2,out,mask)
        return (p_spikes,mask)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =6):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)


        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512*block.expansion, 256)
        self.lif_fc = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)

        self.fc2 = nn.Linear(256, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        output_list = []
        (x,mask) = x
        static_x = self.bn1(self.conv1(x))
        p_spikes = 0
        for t in range(self.total_timestep):
            if t == 0:
                p_spikes =  self.lif_input(static_x)
            else:
                p_spikes = torch.cat((p_spikes,self.lif_input(static_x)),dim=0)
        p_spikes,_ = self.layer1((p_spikes,mask))
        p_spikes,_ = self.layer2((p_spikes,mask))
        p_spikes,_ = self.layer3((p_spikes,mask))
        p_spikes = self.avgpool(p_spikes)
        p_spikes = torch.flatten(p_spikes, 1)
        x = self.fc1(p_spikes)
        prob = self.fc2(x).view(self.total_timestep,int(p_spikes.shape[0]/self.total_timestep),10)

        return prob

def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep)

