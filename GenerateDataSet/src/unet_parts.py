""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_sizes=3, strides=1, paddings=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, InSino=False, mid_channels=None, kernel_sizes=3, strides=1, paddings=1):
        super().__init__()
        if InSino:  # in sinogram Domain
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d([1, 3], [1, 2], [0, 1]),
                DoubleConv(in_channels, out_channels, mid_channels=mid_channels, kernel_sizes=kernel_sizes, strides=strides, paddings=paddings)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, mid_channels=mid_channels, kernel_sizes=kernel_sizes,
                           strides=strides, paddings=paddings)
            )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, up_scale_factors=2, mid_channels=None, up_kernel_sizes=3, up_strides=1, up_paddings=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=up_scale_factors, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=mid_channels, kernel_sizes=up_kernel_sizes, strides=up_strides, paddings=up_paddings)


    def forward(self, x1, x2):
        # print(self.up())
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv_Proj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=3, strides=1, paddings=1):
        super(OutConv_Proj, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class OutConv_Img(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=3, strides=1, paddings=1):
        super(OutConv_Img, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
