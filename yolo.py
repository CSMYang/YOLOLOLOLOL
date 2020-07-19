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
            # do it later
            pass

        # dropout
        elif layer_type == "dropout":
            # do it later
            pass

        # connection layer
        elif layer_type == "connected":
            # do it later
            pass

        # detection
        elif layer_type == "detection":
            # do it later
            pass

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
            if type in ["convolutional", "maxpool"]:
                output = module(output)
            elif type == "local": # dropout, connected, detection
                pass
        return output
