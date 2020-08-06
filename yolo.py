"""
Yolo v1 object detection model.
"""
import torch.nn as nn
from util import get_model_from_config


class Flatten(nn.Module):
    """
    Flatten the result to one dimensional.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Transfer x to one dimentional output.
        """
        return x.view(x.size(0), -1)


def build_yolonet(module_params):
    """
    Construct a Yolo convolutional neural network based on the parameters from the module_params.

    Note: We get the idea of this design from https://github.com/AyushExel/Detectx-Yolo-V3.
    """
    # get the hyperparameters of the neural net.
    net_param = module_params.pop(0)
    channels = [int(net_param['channels'])]
    modules = nn.ModuleList()
    detect_param = module_params.pop(-1)

    # hyperparameters:
    momentum = float(net_param['momentum'])

    # detection layer

    for i, layer in enumerate(module_params):
        module = nn.Sequential()
        layer_type = layer['type']
        # convolutional layer
        if layer_type == "convolutional":
            kernel = int(layer['size'])
            out_channel = int(layer['filters'])
            stride = int(layer['stride'])
            # pad = int(layer['pad'])
            pad = (kernel - 1) // 2
            bn = int(layer['batch_normalize']
                     ) if "batch_normalize" in layer else 0
            # print(type(kernel), type(out_channel), type(stride), type(pad), type(bn))

            conv_layer = nn.Conv2d(in_channels=channels[-1], out_channels=out_channel,
                                   kernel_size=kernel, stride=stride, padding=pad, bias=not bn)
            module.add_module("conv_layer_{}".format(i), conv_layer)
            if bn:
                batch_norm = nn.BatchNorm2d(
                    num_features=out_channel, momentum=momentum)
                module.add_module("batch_norm_{}".format(i), batch_norm)
            if "activation" in layer and layer["activation"] == "leaky":
                leaky = nn.LeakyReLU(negative_slope=0.1, inplace=False)
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
            stride = int(layer['stride'])
            pad = int(layer['pad'])
            out_channel = int(layer['filters']) * \
                          (kernel + pad) * (kernel + pad)
            gird_size = int(2 * kernel + stride)

            # local_layer = Local(in_channels=channels[-1], out_channels=out_channel,
            #                        kernel_size=kernel, stride=stride, padding=pad)
            module.add_module("Flatten_{}".format(i), Flatten())
            local_layer = nn.Linear(in_features=(
                    gird_size * gird_size * channels[-1]), out_features=out_channel)
            module.add_module("local_layer_{}".format(i), local_layer)
            if "activation" in layer and layer["activation"] == "leaky":
                leaky = nn.LeakyReLU(negative_slope=0.1, inplace=False)
                module.add_module("leaky_relu_{}".format(i), leaky)
            # update channels
            channels.append(out_channel)

        # dropout
        elif layer_type == "dropout":
            prob = float(layer['probability'])
            dropout = nn.Dropout(p=prob, inplace=False)
            module.add_module("dropout_{}".format(i), dropout)

        # connection layer
        elif layer_type == "connected":
            out_channel = int(layer['output'])
            if "activation" in layer and layer["activation"] == "linear":
                linear = nn.Linear(
                    in_features=channels[-1], out_features=out_channel)
                module.add_module("linear_{}".format(i), linear)
            sigmoid_layer = nn.Sigmoid()
            module.add_module("sigmoid_{}".format(i), sigmoid_layer)

        # skip the detection layer (detection_cuda will do detection)
        elif layer_type == "detection":  # like yolo layer in yolo-v3
            continue

        # add module
        modules.append(module)
    # print(type(modules))
    return net_param, detect_param, modules


class YoloNet(nn.Module):
    """
    YOLO v1 model.
    """

    def __init__(self, config_file):
        super(YoloNet, self).__init__()
        self.module_params = get_model_from_config(config_file)
        self.hyperparams, self.detection_param, self.m = build_yolonet(
            self.module_params)
        self.img_size = int(self.hyperparams['height'])

    def forward(self, x):
        """
        The forward method for YoloNet.
        """
        # print(x.shape)
        output = x
        for module in self.m:
            output = module(output)
        classes = int(self.detection_param['classes'])
        coords = int(self.detection_param['coords'])
        side = int(self.detection_param['side'])
        num = int(self.detection_param['num'])
        output = output.view(-1, side, side, (1 + coords) * num + classes)
        # print(output)
        return output
