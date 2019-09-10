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

