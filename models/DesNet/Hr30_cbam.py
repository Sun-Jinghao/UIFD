import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from .attentions import CBAMBlock

class AdaptiveFM(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(AdaptiveFM, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
    def forward(self, x):
        return self.conv(x) + x

class HrNet(nn.Module):
    """ Takes a list of images as input, and returns for each image:
            - a pixelwise descriptor
    """

    def __init__(self, output_dim=128, input_chan=3):
        super(HrNet, self).__init__()

        self.x1down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x2down = nn.MaxPool2d(kernel_size=4, stride=4)
        self.preupsoft0 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.preupsoft1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)

        self.cbam1 = CBAMBlock(32,4,7)
        self.cbam2 = CBAMBlock(64,4,7)

        self.stage0_layer0 = nn.Sequential(  # 3,16
            nn.Conv2d(input_chan, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False)
        )
        self.stage0_layer1 = nn.Sequential(  # 3,16
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False)
        )
        self.stage0_layer2 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False)
        )
        self.stage0_layer3 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False)
        )
        self.stage1_layer0 = nn.Sequential(  # 3,16
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False)
        )
        self.stage1_layer1 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False)
        )
        self.stage1_layer2 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False)
        )
        self.stage2_layer0 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False)
        )
        self.desc_head = nn.Sequential(  # 3,16
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=False),
            nn.Conv2d(192, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )



    def forward(self, x):
        feat0 = self.stage0_layer0(x)
        n0, c0, h0, w0 = feat0.shape
        feat1 = self.x1down(feat0)
        feat0 = self.cbam1(feat0)
        feat0 = self.stage0_layer1(feat0)
        feat1 = self.stage1_layer0(self.cbam1(feat1))
        feat1_up = F.interpolate(self.preupsoft0(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0 = self.stage0_layer2(self.cbam2(torch.cat((feat0, feat1_up), dim=1)))
        feat1 = self.stage1_layer1(self.cbam2(torch.cat((feat1, feat0_down), dim=1)))
        del feat1_up, feat0_down
        feat1_up = F.interpolate(self.preupsoft1(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0_down_down = self.x2down(feat0)
        feat1_down = self.x1down(feat1)
        feat0 = self.stage0_layer3(self.cbam2(torch.cat((feat0, feat1_up), dim=1)))
        feat1 = self.stage1_layer2(self.cbam2(torch.cat((feat1, feat0_down), dim=1)))
        del feat1_up, feat0_down
        feat2 = self.stage2_layer0(self.cbam2(torch.cat((feat0_down_down, feat1_down), dim=1)))
        del feat0_down_down, feat1_down
        feat0 = self.x2down(feat0)
        feat1 = self.x1down(feat1)
        x = self.desc_head(torch.cat((feat0, feat1, feat2), dim=1)) # scale = 0.25x
        outputs = {
            'raw_descs_fine': x
        }
        return outputs
