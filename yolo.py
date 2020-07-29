"""
Yolo v1 object detection model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from util import get_model_from_config
from torch.nn.modules.utils import _pair


class Conv2dLocal(nn.Module):
    """
    Construct a local layer for YOLO CNN.
    https://discuss.pytorch.org/t/locally-connected-layers/26979/2
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dLocal, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        fold_num = (in_channels + 2 * padding - self.kernel_size) // self.stride + 1
        self.weight = nn.Parameter(torch.randn(fold_num, kernel_size, out_channels))

    def forward(self, x):
        """
        https://github.com/pytorch/pytorch/issues/499#issuecomment-503962218
        """
        x = F.pad(x, [self.padding] * 2, value=0)
        x = x.unfold(-1, size=self.kernel_size, step=self.stride)
        x = torch.matmul(x.unsqueeze(2), self.weight).squeeze(2)
        return x
    #     self.weight = nn.Parameter(
    #         torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
    #     )
    #
    # def forward(self, x):
    #     """
    #     not finished. implement it later.
    #     """
    #     _, c, h, w = x.size()
    #     kh, kw = self.kernel_size
    #     dh, dw = self.stride
    #     x = x.unfold(2, kh, dh).unfold(3, kw, dw)
    #     x = x.contiguous().view(*x.size()[:-2], -1)
    #     # Sum in in_channel and kernel_size dims
    #     out = (x.unsqueeze(1) * self.weight).sum([2, -1])
    #     if self.bias is not None:
    #         out += self.bias
    #     return out


class Detection(nn.Module):
    """
    Construct a detection layer for YOLO CNN.
    Not finished yet
    """
    def __init__(self, classes, coords, rescore, side, num, softmax, sqrt, jitter, object_s, noobject_s, class_s, coord_s):
        super(Detection, self).__init__()
        self.classes = classes
        self.coords = coords
        self.rescore = rescore
        self.side = _pair(side)
        self.n = num
        self.softmax = softmax
        self.sqrt = sqrt
        self.jitter = jitter
        self.object_scale = object_s
        self.noobject_scale = noobject_s
        self.class_scale = class_s
        self.coord_scale = coord_s

        self.inputs = side*side*((1 + coords) * num + classes)
        self.truths = side*side*(1 + coords + classes)

    def forward(self, x):
        pass


def build_yolonet(module_params):
    """
    Construct a Yolo convolutional neural network based on the parameters from the module_params.
    """
    # get the hyperparameters of the neural net.
    net_param = module_params.pop(0)
    channels = [net_param['channels']]
    modules = nn.ModuleList()

    for i, layer in enumerate(net_param):
        module = nn.Sequential()
        layer_type = layer['type']
        # convolutional layer
        if layer_type == "convolutional":
            kernel = int(layer['size'])
            out_channel = int(layer['filters'])
            stride = int(layer['stride'])
            pad = (kernel - 1) // 2
            bn = int(layer['batch_normalize']) if "batch_normalize" in layer else 0

            conv_layer = nn.Conv2d(in_channels=channels[-1], out_channels=out_channel,
                                   kernel_size=kernel, stride=stride, padding=pad, bias=not bn)
            module.add_module("conv_layer_{}".format(i), conv_layer)
            if bn:
                batch_norm = nn.BatchNorm2d(out_channel)
                module.add_module("batch_norm_{}".format(i), batch_norm)
            if "activation" in layer and layer["activation"] == "leaky":
                leaky = nn.LeakyReLU(negative_slope=0.1)
                module.add_module("leaky_relu_{}".format(i), leaky)
            # update channels
            channels.append(out_channel)

        # max-pooling layer
        elif layer_type == "maxpool":
            kernel = int(layer['size'])
            stride = int(layer['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel, stride=stride)
            module.add_module("max_pool_{}".format(i), maxpool)

        # local
        elif layer_type == "local":
            kernel = int(layer['size'])
            out_channel = int(layer['filters'])
            stride = int(layer['stride'])
            pad = (kernel - 1) // 2

            local_layer = Conv2dLocal(in_channels=channels[-1], out_channels=out_channel,
                                   kernel_size=kernel, stride=stride, padding=pad)
            module.add_module("local_layer_{}".format(i), local_layer)
            if "activation" in layer and layer["activation"] == "leaky":
                leaky = nn.LeakyReLU(negative_slope=0.1)
                module.add_module("leaky_relu_{}".format(i), leaky)
            # update channels
            channels.append(out_channel)

        # dropout
        elif layer_type == "dropout":
            prob = float(layer['probability'])
            dropout = nn.Dropout(p=prob)
            module.add_module("dropout_{}".format(i), dropout)

        # connection layer
        elif layer_type == "connected":
            out_channel = int(layer['output'])
            if "activation" in layer and layer["activation"] == "linear":
                linear = nn.Linear(in_features=channels[-1], out_features=out_channel)
                module.add_module("linear_{}".format(i), linear)

        # detection
        elif layer_type == "detection": # like yolo layer in yolo-v3
            classes = int(layer['classes'])
            coords = int(layer['coords'])
            rescore = int(layer['rescore'])
            side = int(layer['side'])
            num = int(layer['num'])
            softmax = int(layer['softmax'])
            sqrt = int(layer['sqrt'])
            jitter = float(layer['jitter'])
            object_s = int(layer['object_scale'])
            noobject_s = float(layer['noobject_scale'])
            class_s = int(layer['class_scale'])
            coord_s = int(layer['coord_scale'])
            detect = Detection()
            module.add_module("detection", detect)

        # add module
        modules.append(module)
    return net_param, modules


class YoloNet(nn.Module):
    """
    YOLO v1 model.
    """
    def __init__(self, config_file):
        super(YoloNet, self).__init__()
        self.module_params = get_model_from_config(config_file)
        self.hyperparams, self.modules = build_yolonet(self.module_params)
        self.header = None # do it later
        self.seen = 0

    def forward(self, x, cuda=True, train_mode=True):
        """
        The forward method for YoloNet.
        cuda: True if we use gpu for computation
        train_mode: True if it is for training
        """
        output = x
        for params, module in zip(self.module_params[1:], self.modules):
            type = params["type"]
            if type in ["convolutional", "maxpool", "local", "dropout", "connected"]:
                output = module(output)
            elif type == "detection": # detection
                # do it later
                pass
        return output
