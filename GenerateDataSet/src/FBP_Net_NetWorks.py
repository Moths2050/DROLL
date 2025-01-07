'''
this network is test using TransportConv vs interpolate
using the old network without any changes
two U-Net and let fbp-layer conj them
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from src import FBP_Layer
# from src import Spatial2FrequencyLayer
# from src import Frequency2SpatialLayer
from src import unet_parts

class FBP_Net(nn.Module):
    def __init__(self, args, out_ch_pre=96, out_ch_post=96):
        super(FBP_Net, self).__init__()

        self.Sino_In = unet_parts.DoubleConv(1, 64, mid_channels=None, kernel_sizes=[1, 3], strides=[1, 1], paddings=[0, 1])

        self.Sino_Down1 = unet_parts.Down(64, 128, mid_channels=None, InSino=True, kernel_sizes=[1, 3], strides=[1, 1], paddings=[0, 1])
        self.Sino_Down2 = unet_parts.Down(128, 256, mid_channels=None, InSino=True, kernel_sizes=[1, 3], strides=[1, 1], paddings=[0, 1])
        self.Sino_Down3 = unet_parts.Down(256, 512, mid_channels=None, InSino=True, kernel_sizes=[1, 3], strides=[1, 1], paddings=[0, 1])
        self.Sino_Down4 = unet_parts.Down(512, 512, mid_channels=None, InSino=True, kernel_sizes=[1, 3], strides=[1, 1], paddings=[0, 1])

        self.Sino_Up1 = unet_parts.Up(1024, 256, bilinear=True, up_scale_factors=(1, 2), mid_channels=None)
        self.Sino_Up2 = unet_parts.Up(512, 128, bilinear=True, up_scale_factors=(1, 2), mid_channels=None)
        self.Sino_Up3 = unet_parts.Up(256, 64, bilinear=True, up_scale_factors=(1, 2), mid_channels=None)
        self.Sino_Up4 = unet_parts.Up(128, 64, bilinear=True, up_scale_factors=(1, 2), mid_channels=None)

        self.Sino_Out = unet_parts.OutConv_Proj(64, 1, kernel_sizes=[1, 1], strides=[1, 1], paddings=[0, 0])

        #### 2.
        self.FBP = FBP_Layer.FBP_Layer(args)

        #### 3.
        self.Img_In = unet_parts.DoubleConv(1, 64)

        self.Img_Down1 = unet_parts.Down(64, 128)
        self.Img_Down2 = unet_parts.Down(128, 256)
        self.Img_Down3 = unet_parts.Down(256, 512)
        self.Img_Down4 = unet_parts.Down(512, 512)

        self.Img_Up1 = unet_parts.Up(1024, 256, bilinear=True)
        self.Img_Up2 = unet_parts.Up(512, 128, bilinear=True)
        self.Img_Up3 = unet_parts.Up(256, 64, bilinear=True)
        self.Img_Up4 = unet_parts.Up(128, 64, bilinear=True)

        self.Img_Out = unet_parts.OutConv_Img(64, 1, kernel_sizes=[1, 1], strides=[1, 1], paddings=[0, 0])
        ''' use TransposeConv instead of interpolate    2020-12-24'''
        # self.Img_Up1 = unet_parts.Up(1024, 256, bilinear=False)
        # self.Img_Up2 = unet_parts.Up(512, 128, bilinear=False)
        # self.Img_Up3 = unet_parts.Up(256, 64, bilinear=False)
        # self.Img_Up4 = unet_parts.Up(128, 64, bilinear=False)
        #
        # self.Img_Out = unet_parts.OutConv_Img(64, 1, kernel_sizes=[1, 1], strides=[1, 1], paddings=[0, 0])

        ################################################################

    def forward(self, x):
        # Sinogram Domain
        # print(x.shape)
        Sino1 = self.Sino_In(x)
        # print(Sino1.shape)
        Sino2 = self.Sino_Down1(Sino1)
        # print(Sino2.shape)
        Sino3 = self.Sino_Down2(Sino2)
        # print(Sino3.shape)
        Sino4 = self.Sino_Down3(Sino3)
        # print(Sino4.shape)
        Sino5 = self.Sino_Down4(Sino4)
        # print(Sino5.shape)
        SinoOut = self.Sino_Up1(Sino5, Sino4)
        # print(SinoOut.shape)
        SinoOut = self.Sino_Up2(SinoOut, Sino3)
        # print(SinoOut.shape)
        SinoOut = self.Sino_Up3(SinoOut, Sino2)
        # print(SinoOut.shape)
        SinoOut = self.Sino_Up4(SinoOut, Sino1)
        # print(SinoOut.shape)
        SinoOut = self.Sino_Out(SinoOut)
        # print(SinoOut.shape)

        # BP Layer
        Img = self.FBP(SinoOut)

        # image domain
        Img1 = self.Img_In(Img)
        # print(Img1.shape)
        Img2 = self.Img_Down1(Img1)
        # print(Img2.shape)
        Img3 = self.Img_Down2(Img2)
        # print(Img3.shape)
        Img4 = self.Img_Down3(Img3)
        # print(Img4.shape)
        Img5 = self.Img_Down4(Img4)
        # print(Img5.shape)
        ImgOut = self.Img_Up1(Img5, Img4)
        # print(ImgOut.shape)
        ImgOut = self.Img_Up2(ImgOut, Img3)
        # print(ImgOut.shape)
        ImgOut = self.Img_Up3(ImgOut, Img2)
        # print(ImgOut.shape)
        ImgOut = self.Img_Up4(ImgOut, Img1)
        # print(ImgOut.shape)
        ImgOut = self.Img_Out(ImgOut)
        # print(ImgOut.shape)

        return ImgOut
