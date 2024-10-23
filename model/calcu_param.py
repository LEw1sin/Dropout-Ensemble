
import torch
from thop import profile, clever_format
from DE_framework import DE_framework
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input = torch.randn(4, 1, 224, 224).to(device)

weight_list = [0.33,0.33,0.33]
model = DE_framework(models=[UNet(num_classes = 4, max_channels=256,),
                            UNet(num_classes = 4, max_channels=256,),
                            UNet(num_classes = 4, max_channels=256,)],
                                        weight_list=weight_list).to(device)

flops, params = profile(model, inputs=(input, ))
flops, params = clever_format([flops, params], "%.3f")

print(f'Flops: {flops}')
print(f'Params: {params}')



