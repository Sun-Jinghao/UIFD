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
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(inplace=True)
        )

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        #self.layer2 = CBAMBlock(channel=32,reduction=2,kernel_size=3)
        self.layer2 = nn.Sequential(#16，16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(16, affine=False),
            nn.ReLU(inplace=True)
        )
        #up1 32 32
        self.up1 = nn.ConvTranspose2d(16,16 , kernel_size=4, stride=2, padding=1)

        self.layer3 = nn.Sequential(# 128 128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True)
        )
        #up1 128 128 
        self.up2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        #self.layer4 = CBAMBlock(channel=256,reduction=4,kernel_size=3)

        self.layer4 = nn.Sequential(#64 64
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.InstanceNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=8, dilation=16)
        )
        # up3 
        self.up3 = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, output_dim, kernel_size=2, stride=1, padding=8, dilation=16),
            nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=8, dilation=16)
            
        )
       # 
        self.up4 = torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
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

class AdaptiveFM(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(AdaptiveFM, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
    def forward(self, x):
        return self.conv(x) + x


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self,out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel, affine=False)
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel*2, affine=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y



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

        self.stage0_layer0 = ResBlock_do_fft_bench(32)
        self.stage0_layer1 = ResBlock_do_fft_bench(32)
        self.stage0_layer2 = ResBlock_do_fft_bench(64)#
        self.stage0_layer3 = ResBlock_do_fft_bench(64)
        self.stage1_layer0 = ResBlock_do_fft_bench(32)
        self.stage1_layer1 = ResBlock_do_fft_bench(64)#
        self.stage1_layer2 = ResBlock_do_fft_bench(64)
        self.stage2_layer0 = ResBlock_do_fft_bench(64)
        self.desc_head = ResBlock_do_fft_bench(192)
        self.conv1 = nn.Sequential(  # 3,16
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv22 = nn.Sequential(  # 3,16
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv32 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=False, groups=1))
        self.conv64 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=False, groups=1))
        self.cbam1 = CBAMBlock(32,4,7)
        self.cbam2 = CBAMBlock(64,4,7)
 

    def forward(self, x):
        feat0 = self.conv1(x)
        feat0 = self.stage0_layer0(feat0)
        n0, c0, h0, w0 = feat0.shape
        feat1 = self.x1down(feat0)
        feat0 = self.stage0_layer1(feat0)
        feat1 = self.stage1_layer0(self.conv32(self.cbam1(feat1)))
        feat1_up = F.interpolate(self.preupsoft0(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0 = self.stage0_layer2(torch.cat((feat0, feat1_up), dim=1))
        feat0 = self.conv22(feat0)
        feat1 = self.stage1_layer1(self.cbam2(torch.cat((feat1, feat0_down), dim=1)))
        feat1 = self.conv22(feat1)
        del feat1_up, feat0_down
        feat1_up = F.interpolate(self.preupsoft1(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0_down_down = self.x2down(feat0)
        feat1_down = self.x1down(feat1)
        feat0 = self.stage0_layer3(torch.cat((feat0, feat1_up), dim=1))
        feat1 = self.stage1_layer2(self.cbam2(self.con64(torch.cat((feat1, feat0_down), dim=1))))
        del feat1_up, feat0_down
        feat2 = self.stage2_layer0(self.cbam2(self.con64(torch.cat((feat0_down_down, feat1_down), dim=1)))
        del feat0_down_down, feat1_down
        feat0 = self.x2down(feat0)
        feat1 = self.x1down(feat1)
        x = self.desc_head(torch.cat((feat0, feat1, feat2), dim=1)) # scale = 0.25x
        outputs = {
            'raw_descs_fine': x
        }
        return outputs

