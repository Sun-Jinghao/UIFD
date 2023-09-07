import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
# input:feature  output:feature

#SE module
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# efficient channel attention
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # b c 1 1

        # Two different branches of ECA module #(b,c,1,1)->(b,c,1)->(b,1,c)->(b,1,c)->(b,c,1)->(b,c,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y) #(b,c,1,1)

        return x * y.expand_as(x) #(b,c,h,w).*(b,c,h,w)->(b,c,h,w)

#External attention
#from https://github.com/MenghaoGuo/-EANet/blob/main/model_torch.py

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, n)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

#Coordinate Attention
#https://github.com/Andrew-Qibin/CoordAttention
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        F.adaptive_avg_pool2d()

        return out

# A Simple, Parameter-Free Attention Module
#https://github.com/ZjjConan/SimAM

class SimAM_Block(nn.Module):
    def __init__(self, lambda_=1e-4):
        super(SimAM_Block, self).__init__()
        self.lambda_ = lambda_
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # spatial size
        n = w*h - 1
        # square of (t - u)
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.lambda_)) + 0.5
        # return attended features
        return x * self.activation(E_inv)

class HieraSimAM_Block(nn.Module):
    def __init__(self, lambda_=1e-4):
        super(HieraSimAM_Block, self).__init__()
        self.lambda_ = lambda_
        self.activation = nn.Sigmoid()
    def forward(self, x):
        n, c, h, w = x.size()
        n = w * h
        # mean_1 = F.adaptive_avg_pool2d(x, 1)
        # mean_1 = F.interpolate(mean_1,size = (h,w),mode='nearest')
        # mean_2 = F.adaptive_avg_pool2d(x, 2)
        # mean_2 = F.interpolate(mean_2, size=(h, w), mode='nearest')
        # mean_3 = F.adaptive_avg_pool2d(x, 4)
        # mean_3 = F.interpolate(mean_3, size=(h, w), mode='nearest')
        # mean_4 = F.adaptive_avg_pool2d(x, 8)
        # mean_4 = F.interpolate(mean_4, size=(h, w), mode='nearest')

        mean_4 = F.adaptive_avg_pool2d(x, 8)
        mean_3 = F.adaptive_avg_pool2d(mean_4, 4)
        mean_2 = F.adaptive_avg_pool2d(mean_3, 2)
        mean_1 = F.adaptive_avg_pool2d(mean_2, 1)
        mean_1 = F.interpolate(mean_1, size=(h, w), mode='nearest')
        mean_2 = F.interpolate(mean_2, size=(h, w), mode='nearest')
        mean_3 = F.interpolate(mean_3, size=(h, w), mode='nearest')
        mean_4 = F.interpolate(mean_4, size=(h, w), mode='nearest')
        #局部区域显著性提取
        #max_mean = torch.max(torch.max(mean_1, mean_2), torch.max(mean_3, mean_4))
        max_mean = torch.stack([mean_1, mean_2, mean_3, mean_4], dim=1)
        max_mean, _ = torch.max(max_mean, dim=1)
        #全局偏差
        d = (x - max_mean).pow(2)
        #计算每一细分小块的方差并求平均 8*8块
        v = d.sum(dim=[2, 3], keepdim=True) / (n-64)
        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.lambda_)) + 0.5
        # return attended features
        return x * self.activation(E_inv)

#GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

#提速版本
#使用之前xxx = torch.jit.script(Fused_GELU())
class Fused_GELU(nn.Module):
    def __init__(self):
        super(Fused_GELU, self).__init__()
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

class My_GELU(nn.Module):
    def __init__(self):
        super(My_GELU, self).__init__()
        self.func_gelu = torch.jit.script(Fused_GELU())
    def forward(self, x):
        return self.func_gelu(x)
