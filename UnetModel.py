import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 # np.array -> torch.tensor
import os
from glob import glob
import timm

def unet_block(in_channels,out_channels):
  return nn.Sequential(
      nn.Conv2d(in_channels,out_channels,3,1,1),
      nn.ReLU(),
      nn.Conv2d(out_channels,out_channels,3,1,1),
      nn.ReLU(),
  )


class Unet(nn.Module):
  def __init__(self, n_classes):
      super().__init__()
      self.n_classes = n_classes
      self.backbone = timm.create_model("efficientnet_b0",pretrained=True,features_only=True)
      self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
      self.block_neck = unet_block(320,112)
      self.block_up1 = unet_block(112+112,40)
      self.block_up2 = unet_block(40+40,24)
      self.block_up3 = unet_block(24+24,16)
      self.block_up4 = unet_block(16+16,16)
      self.conv_cls = nn.Conv2d(16,self.n_classes,1)

  def forward(self,x):
    x1,x2,x3,x4,x5 = self.backbone(x) 
    x = self.block_neck(x5) #(B,112,8,8)
    x = torch.cat([self.upsample(x),x4], dim=1)
    x = self.block_up1(x)   #(B,40,16,16)

    x = torch.cat([self.upsample(x),x3], dim=1)
    x = self.block_up2(x)#(B,24,32,32)

    x = torch.cat([self.upsample(x),x2], dim=1)
    x = self.block_up3(x)#(B,16,64,64)

    x = torch.cat([self.upsample(x),x1], dim=1)
    x = self.block_up4(x)#(B,16,128,128)
    x = self.conv_cls(x) #(B,21,128,128)
    x = self.upsample(x) #(B,21,256,256)
    return x