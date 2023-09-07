import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
from attentions import CBAMBlock
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
 
 
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.x1down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x2down = nn.MaxPool2d(kernel_size=4, stride=4)
        self.preupsoft0 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.preupsoft1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.cbam1 = CBAMBlock(32,4,7)
        self.cbam2 = CBAMBlock(64,4,7)
        self.stage0_layer0 = nn.Sequential(  # 3,16
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
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
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
 
    def forward(self, x):
        outputs = {}
        # for name, module in self.submodule._modules.items():
        #     if "fc" in name:
        #         x = x.view(x.size(0), -1)
        #
        #     x = module(x)
        #     print(name)
        #     if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
        #         outputs[name] = x
 
        feat0 = self.stage0_layer0(x)
        n0, c0, h0, w0 = feat0.shape
        feat1 = self.x1down(feat0)

        feat0 = self.cbam1(feat0)

        feat0 = self.stage0_layer1(feat0)
        feat1 = self.stage1_layer0(feat1)
        feat1_up = F.interpolate(self.preupsoft0(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0 = self.stage0_layer2(torch.cat((feat0, feat1_up), dim=1))
        feat1 = self.stage1_layer1(torch.cat((feat1, feat0_down), dim=1))
        del feat1_up, feat0_down
        feat1_up = F.interpolate(self.preupsoft1(feat1), size=[h0, w0], mode='bilinear', align_corners=True)
        feat0_down = self.x1down(feat0)
        feat0_down_down = self.x2down(feat0)
        feat1_down = self.x1down(feat1)
        feat0 = self.stage0_layer3(self.cbam2(torch.cat((feat0, feat1_up), dim=1)))
        feat1 = self.stage1_layer2(torch.cat((feat1, feat0_down), dim=1))
        del feat1_up, feat0_down
        feat2 = self.stage2_layer0(torch.cat((feat0_down_down, feat1_down), dim=1))
        del feat0_down_down, feat1_down
        feat0 = self.x2down(feat0)
        feat1 = self.x1down(feat1)
        x = self.desc_head(torch.cat((feat0, feat1, feat2), dim=1)) # scale = 0.25x
################修改成自己的网络，直接在network.py中return你想输出的层
        # stage0_layer0,stage0_layer1,stage0_layer2,stage0_layer3,stage1_layer0,stage1_layer0,stage1_layer2,stage2_layer0,desc_head
        # #x1,x2,x3,x4,x5,x6,up7,merge7,conv7,up8,merge8,conv8,up9,merge9,conv9,up10,merge10,conv10,up11,merge11,conv11,conv12,mask,x2_0 = self.submodule(x)
        # outputs["stage0_layer0"] = stage0_layer0
        # outputs["stage0_layer1"] = stage0_layer1
        # outputs["stage0_layer2"] = stage0_layer2
 
        # outputs["stage0_layer3"] = stage0_layer3
        # outputs["stage1_layer0"] = stage1_layer0
        # outputs["stage1_layer0"] = stage1_layer0
 
        # outputs["stage1_layer2"] =stage1_layer2
        # outputs["stage2_layer0"] = stage2_layer0
        # outputs["desc_head"] = desc_head
 
 
        outputs.append(x)
 
        # return outputs
        return outputs
 
 
def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)
 
 
def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
 
 
def get_feature():
    pic_dir = '1.jpg' #往网络里输入一张图片
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)
 
    img = img.to(device)
 
    net = torch.load('/home/ouc/zh/Documents/experiments/Desc_Hr_paper/checkpoints/checkpoint_837000.pth')
    exact_list = None
    # exact_list = ['conv1_block',""]
    dst = './features' #保存的路径
    therd_size = 256 #有些图太小，会放大到这个尺寸
 
    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue
 
            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
 
            dst_path = os.path.join(dst, k)
 
            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
 
            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)
 
 
if __name__ == '__main__':
    get_feature()
 