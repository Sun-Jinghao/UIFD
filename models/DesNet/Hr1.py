import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from .attentions import CBAMBlock

class Hr(nn.Module):
    """ Takes a list of images as input, and returns for each image:
            - a pixelwise descriptor
    """
    def __init__(self, output_dim=128, input_chan=3):
        super(Hr, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_chan, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True)
        )

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        #self.layer2 = CBAMBlock(channel=32,reduction=2,kernel_size=3)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(32,32 , kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(output_dim, affine=False),
            nn.ReLU(inplace=True)
        )
        self.up2 = torch.nn.ConvTranspose2d(128,128 , kernel_size=4, stride=2, padding=1)
        #self.layer4 = CBAMBlock(channel=256,reduction=4,kernel_size=3)

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.InstanceNorm2d(256, affine=False),
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=8, dilation=16)
        )
        self.up3 = torch.nn.ConvTranspose2d(256,256 , kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(768, output_dim, kernel_size=2, stride=1, padding=8, dilation=16),
            nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=8, dilation=16)
            #768
        )


    def _forw_impl(self, x):
        x1 = self.layer1(x)
        down01 = self.down(x1)
        x2 = self.layer2(x1)
        #print("*******************",x2.shape)
        up1 = self.up1(down01)
        #print("*******************",up1.shape)

        del x1
     
        x21 = self.layer2(down01) 
        del down01
        down02 = self.down(x2)
        x3 = self.layer3(torch.cat((x2,up1),1))
        x31 = self.layer3(torch.cat((down02,x21),1))
        del x2,up1,x21

        up2 = self.up2(x31)
        x4 = self.layer4(torch.cat((x3,up2),1))
        
        down03 = self.down(x3)
        x41 = self.layer4(torch.cat((down03,x31),1))

        d3 = self.down(down03)

        down13 = self.down(x31)
        del x31
        x410 = self.layer4(torch.cat((down13,d3),1))

        up3 = self.up3(x41)
        up31 = self.up3(self.up3(x410))
        
        x5 = torch.cat((x4,up3),1)
        x5 = torch.cat((x5,up31),1)

        x = self.layer5(x5)
        return x

    def forward(self, x):

        x = self._forw_impl(x)
        outputs = {
            'raw_descs_fine': x
        }


        return outputs