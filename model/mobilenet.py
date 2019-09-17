import os 
import argparse
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data 


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # Define standard convolutional layer with batchnorm
        # as defined in paper
        # | 3 x 3 conv | --> | BN | --> |RELU| | 
        # input , output, stride

        def conv_bn(inp, out, stride):

            return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bisa=False), 
                    nn.BatchNorm2(oup), 
                    nn.ReLU(inplace=True)
                    )

        # | BN | --> | ReLU | --> | 1x1 Conv | --> BatchNorm  --> | ReLU|
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), 
                    nn.BatchNorm2d(inp), 
                    nn.ReLU(inplace=True), 
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                    nn.BatchNorm(oup), 
                    nn.ReLU(inplace=True),
                    )

        def conv_1x1_bn(inp, oup, onnx_compatible=False):
            ReLU =  nn.ReLU if onnx_compatible else nn.ReLU6
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup),
                ReLU(inplace=True)
            )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, onnx_compatible=False):
        """
        params: 
            inp: input 
            oup: output
            stride: Stride size
            expand_ratio: 
        """
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        # Stride check 
        assert stride in [1,2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ReLU(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2(inp, hidden_dim, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                ReLU(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup), 
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1, input_size = 224, width_mult=1., dropout_ratio=0.1,
    onnx_compatible=False):
        super(MobileNetV2).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 2], 
        [6, 32, 3, 2], 
        [6, 64, 4, 2], 
        [6, 96, 3, 1], 
        [6, 160, 3, 2], 
        [6, 320, 1, 1], 
        ]

    # Building layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, 
                        s, expand_ratio=t, onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1, 
                        expand_ratio=t, onnx_compatible=onnx_compatible))
                input_channel=output_channel

            self.features.append(conv_1x1_bn(input_channel, self.last_channel, 
                onnx_compatible=onnx_compatible))
            self.features = nn.Sequential(*self.features)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_ratio), 
                nn.Linear(self.last_channel, n_class), 
            )
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, matg.sqrt(2. /n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()