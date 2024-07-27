import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from spikingjelly.clock_driven import surrogate, neuron

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, total_timestep=5):
        super(AlexNet, self).__init__()

        self.total_timestep = total_timestep
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.lif3 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.lif4 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.lif5 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 2 * 2, num_classes)
     
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)


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


    def forward(self, x, mask=0):
        static_x = self.conv1(x)
        p_spikes = 0
        for t in range(self.total_timestep):
            if t == 0:
                p_spikes =  self.lif1(static_x)
            else:
                p_spikes = torch.cat((p_spikes,self.lif1(static_x)),dim=0)

        out = self.maxpool1(p_spikes)
        out = self.conv2(out)
        p_spikes = self._parallel_lif(self.lif2,out,mask)
        out = self.maxpool2(p_spikes)
        out = self.conv3(p_spikes)
        p_spikes = self._parallel_lif(self.lif3,out,mask)
        out = self.conv4(p_spikes)
        p_spikes = self._parallel_lif(self.lif4,out,mask)
        out = self.conv5(p_spikes)
        p_spikes = self._parallel_lif(self.lif5,out,mask)
        out = self.maxpool3(p_spikes)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        prob = self.fc1(out).view(self.total_timestep,int(p_spikes.shape[0]/self.total_timestep),self.num_classes)
        
        return prob
