import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from .attentions import CBAMBlock

class Hr3(nn.Module):
    """ Takes a list of images as input, and returns for each image:
            - a pixelwise descriptor
    """
    def __init__(self, output_dim=128, input_chan=3):
        super(Hr3, self).__init__()
        
        self.layer1 = nn.Sequential( #3,16
            nn.Conv2d(input_chan, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(inplace=True),
            # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16, affine=False),
            # nn.ReLU(inplace=True)
        )

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        #self.layer2 = CBAMBlock(channel=32,reduction=2,kernel_size=3)
        self.layer2 = nn.Sequential(#16，16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2),
            # nn.InstanceNorm2d(16, affine=False),
            # nn.ReLU(inplace=True)
        )
        #up1 32 32
        self.up1 = nn.ConvTranspose2d(16,16 , kernel_size=4, stride=2, padding=1)

        self.layer3 = nn.Sequential(# 128 128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4),
            # nn.BatchNorm2d(32, affine=False),
            # nn.ReLU(inplace=True)
        )
        #up1 128 128 
        self.up2 = torch.nn.ConvTranspose2d(32,32 , kernel_size=4, stride=2, padding=1)
        #self.layer4 = CBAMBlock(channel=256,reduction=4,kernel_size=3)

        self.layer4 = nn.Sequential(#64 64
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=4, dilation=8),
            # nn.InstanceNorm2d(64, affine=False),
            # nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=8, dilation=16)
        )
        # up3 
        self.up3 = torch.nn.ConvTranspose2d(64,64 , kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, output_dim, kernel_size=2, stride=1, padding=8, dilation=16),
            nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=8, dilation=16)
            
        )
       # 
        self.up4 = torch.nn.ConvTranspose2d(128,128 , kernel_size=4, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(384, output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):

        #x = self._forw_impl(x)
        #1
        x1 = self.layer1(x)
        
        #2
        x2 = self.layer2(x1)
        x2_1 = self.layer2(self.down(x1))

        #3  开始融合
        x3 = self.layer3(torch.cat((x2,self.up1(x2_1)),1))
        x3_1 = self.layer3(torch.cat((self.down(x2),x2_1),1))

        #4 
        x4 = self.layer4(torch.cat((x3,self.up2(x3_1)),1))
        x4_1 = self.layer4(torch.cat((self.down(x3),x3_1),1))
        x4_2 = self.layer4(torch.cat((self.down(self.down(x3)),self.down(x3_1)),1))

        #5
        c = torch.cat((x4,self.up3(x4_1)),1)
        c= torch.cat((c,self.up3(self.up3(x4_2))),1)
        x5 = self.layer5(c)

        c1 = torch.cat((self.down(x4),x4_1),1)
        c1 = torch.cat((c1,self.up3(x4_2)),1)
        x5_1 = self.layer5(c1)

        c2 = torch.cat((self.down(self.down(x4)),self.down(x4_1)),1)
        c2 = torch.cat((c2,x4_2),1)
        x5_2 = self.layer5(c2)

        #6
        x = torch.cat((x5,self.up4(x5_1)),1)
        x = torch.cat((x,self.up4(self.up4(x5_2))),1)
        x = self.layer6(x)
        

        outputs = {
            'raw_descs_fine': x
        }
        return outputs


