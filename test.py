import torch
from model.DE_framework import *
from model.unet import UNet_linear, UNet_logvar
from model.deeplabv3_plus import DeepLabV3P_linear, DeepLabV3P_logvar

import argparse
parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
args = parser.parse_args()
t = torch.rand(4, 10, 1, 224, 224).to(device='cuda:1')
net = DE_framework_mem(args, models=[UNet_logvar(num_classes = 4, max_channels=256),
                                        DeepLabV3P_logvar(num_classes = 4, max_channels=256)]).to(device=t.device)
pred = net(t)
print(pred.size())