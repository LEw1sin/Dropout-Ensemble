import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .tools import uncertainty_weights

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.singleconv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride = 1)
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)        
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,stride=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        return self.conv(x)
                
class UNet_linear(nn.Module):
    def __init__(self, num_classes, bilinear=True, max_channels=256, input_channel=1):
        super(UNet_linear, self).__init__()
        self.n_channels = input_channel
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.max_channels = max_channels

        self.inc = DoubleConv(self.n_channels, self.max_channels//8)
        self.down1 = Down(self.max_channels//8, self.max_channels//4)
        self.down2 = Down(self.max_channels//4, self.max_channels//2)
        self.down3 = Down(self.max_channels//2, self.max_channels)
        self.down4 = Down(self.max_channels, self.max_channels)

        self.dropout = nn.Dropout(0.5)

        self.up1 = Up(self.max_channels*2, self.max_channels//2, self.bilinear)
        self.up2 = Up(self.max_channels, self.max_channels//4, self.bilinear)
        self.up3 = Up(self.max_channels//2, self.max_channels//8, self.bilinear)
        self.up4 = Up(self.max_channels//4, self.max_channels//8, self.bilinear)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.outc = OutConv(self.max_channels//8, self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4)

        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet_logvar(nn.Module):
    def __init__(self, num_classes, bilinear=True, max_channels=256, input_channel=1):
        super(UNet_logvar, self).__init__()
        self.n_channels = input_channel
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.max_channels = max_channels

        self.inc = DoubleConv(self.n_channels, self.max_channels//8)
        self.down1 = Down(self.max_channels//8, self.max_channels//4)
        self.down2 = Down(self.max_channels//4, self.max_channels//2)
        self.down3 = Down(self.max_channels//2, self.max_channels)
        self.down4 = Down(self.max_channels, self.max_channels)
        self.dropout = nn.Dropout(0.5)
        self.up1 = Up(self.max_channels*2, self.max_channels//2, self.bilinear)
        self.up2 = Up(self.max_channels, self.max_channels//4, self.bilinear)
        self.up3 = Up(self.max_channels//2, self.max_channels//8, self.bilinear)
        self.up4 = Up(self.max_channels//4, self.max_channels//8, self.bilinear)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.outc = OutConv(self.max_channels//8, self.num_classes)
        self.uncertainty_weights = uncertainty_weights(self.num_classes, self.max_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        
        x5 = self.dropout(x5)
        uncertainty_weights = self.uncertainty_weights(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, uncertainty_weights


if __name__ == "__main__":
    t = torch.rand(4, 1, 224, 224).to('cuda:5')
    net = UNet_logvar(num_classes = 4, input_channel=1).to(device=t.device)
    pred = net(t)
    print(pred.size())
