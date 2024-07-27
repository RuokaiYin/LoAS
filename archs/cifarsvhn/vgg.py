from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5, total_timestep: int = 6
    ) -> None:
        super().__init__()
        self.total_timestep = total_timestep
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

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

    #* The forward function that is specifically modified for vgg16-bn
    def forward(self, x: torch.Tensor, mask=0) -> torch.Tensor:
        static_x = self.features[:2](x) #? Direct Encoding
        p_spikes = 0
        for t in range(self.total_timestep):
            if t == 0:
                p_spikes =  self.features[2:3](static_x)
            else:
                p_spikes = torch.cat((p_spikes,self.features[2:3](static_x)),dim=0)
        
        #! We are not masking out the spikes from the direct encoder
        # TODO: add support to let the user to configure whether they wanna mask them or not.
        # Should be very easy to be achieved though.

        x = self.features[3:5](p_spikes)
        p_spikes = self._parallel_lif(self.features[5:6],x,mask)
        p_spikes = self.features[6:7](p_spikes)
        
        x = self.features[7:9](p_spikes)
        p_spikes = self._parallel_lif(self.features[9:10],x,mask)
        x = self.features[10:12](p_spikes)
        p_spikes = self._parallel_lif(self.features[12:13],x,mask)
        p_spikes = self.features[13:14](p_spikes)

        x = self.features[14:16](p_spikes)
        p_spikes = self._parallel_lif(self.features[16:17],x,mask)
        x = self.features[17:19](p_spikes)
        p_spikes = self._parallel_lif(self.features[19:20],x,mask)
        x = self.features[20:22](p_spikes)
        p_spikes = self._parallel_lif(self.features[22:23],x,mask)
        p_spikes = self.features[23:24](p_spikes)

        x = self.features[24:26](p_spikes)
        p_spikes = self._parallel_lif(self.features[26:27],x,mask)
        x = self.features[27:29](p_spikes)
        p_spikes = self._parallel_lif(self.features[29:30],x,mask)
        x = self.features[30:32](p_spikes)
        p_spikes = self._parallel_lif(self.features[32:33],x,mask)
        p_spikes = self.features[33:34](p_spikes)

        x = self.features[34:36](p_spikes)
        p_spikes = self._parallel_lif(self.features[36:37],x,mask)
        x = self.features[37:39](p_spikes)
        p_spikes = self._parallel_lif(self.features[39:40],x,mask)
        x = self.features[40:42](p_spikes)
        p_spikes = self._parallel_lif(self.features[42:43],x,mask)
        p_spikes = self.features[43:44](p_spikes)
        p_spikes = torch.flatten(p_spikes, 1)

        prob = self.classifier(p_spikes).view(self.total_timestep,int(p_spikes.shape[0]/self.total_timestep),10)
    
        return prob

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#! The scalable function to create the VGG networks. 
#! If you wanna modify the LIF parameters, here is where you wanna make the changes!!!
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v),
                           neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3., #! Plug in any LIF neuron model.
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
                           ]
            else:
                layers += [conv2d,
                           neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3., #! Plug in any LIF neuron model.
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
                           ]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)