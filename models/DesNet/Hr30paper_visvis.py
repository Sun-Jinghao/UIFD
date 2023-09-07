 
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
 
 
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")
 
 
def create_model():
    model = Sequential()
 
    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
 
    # 第二层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
 
    # 第三层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    # 第四层CNN
    model.add(Convolution2D(9, 3, 3, input_shape=img.shape))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
 
    return model
 
 
if __name__ == "__main__":
    img = cv2.imread('jpg/1.jpg')
 
    model = create_model()
 
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
 
    visualize_feature_map(conv_img)



# import torch.nn as nn
# import math
# import torch.nn.functional as F
# import torch
# from attentions import CBAMBlock
# import os
# import torch
# import torchvision as tv
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# import argparse
# import skimage.data
# import skimage.io
# import skimage.transform
# import numpy as np
# import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Load training and testing datasets.
# pic_dir = 'jpg/7.jpg'
# # 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
# transform = transforms.ToTensor()

# # 单张图像送入
# # 构建网络
# # 提取中间层
# # 可视化特征图

# def get_picture(picture_dir, transform):
#     '''
#     该算法实现了读取图片，并将其类型转化为Tensor
#     '''
#     img = skimage.io.imread(picture_dir)
#     img256 = skimage.transform.resize(img, (256, 256))
#     img256 = np.asarray(img256)
#     img256 = img256.astype(np.float32)

#     return transform(img256)


# def get_picture_rgb(picture_dir):
#     '''
#     该函数实现了显示图片的RGB三通道颜色
#     '''
#     img = skimage.io.imread(picture_dir)
#     img256 = skimage.transform.resize(img, (256, 256))
#     skimage.io.imsave('new4.jpg', img256)

    
#     img = img256.copy()
#     ax = plt.subplot()
#     ax.set_title('new-image')
#     # ax.axis('off')
#     plt.imshow(img)

#     plt.show()
# class L2Net_CBAM(nn.Module):
#     """ Takes a list of images as input, and returns for each image:
#             - a pixelwise descriptor
#     """
#     def __init__(self, output_dim=128, input_chan=3):
#         super(L2Net_CBAM, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(input_chan, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(inplace=True)
#         )
#         self.cbam = CBAMBlock(channel=64,reduction=4,kernel_size=3)
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(128, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, output_dim, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=2, dilation=4),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=4, dilation=8),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=8, dilation=16)
#         )

#     def _forw_impl(self, x):
#         x = self.layer1(x)
#         x = self.cbam(x)
#         x = self.layer2(x)
#         return x

#     def _forw_test(self, x):
#         x = self.layer(x)
#         #x = F.normalize(x, p=2, dim=1)
#         return x

#     def forward(self, x):
#         x = self._forw_impl(x)

#         #print("*******************outputs.shape",outputs['raw_descs_fine'].shape)

#         return x

# class L2Net(nn.Module):
#     """ Takes a list of images as input, and returns for each image:
#             - a pixelwise descriptor
#     """
#     def __init__(self, output_dim=128, input_chan=3):
#         super(L2Net, self).__init__()

#         self.layer = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(128, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, output_dim, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=2, dilation=4),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=4, dilation=8),
#             nn.BatchNorm2d(output_dim, affine=False),
#             nn.Conv2d(output_dim, output_dim, kernel_size=2, stride=1, padding=8, dilation=16)
#         )

#     def forward(self, x):
#         x = self.layer(x)
#         return x

# # 中间特征提取
# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = extracted_layers
 
#     def forward(self, x):
#         outputs = []
#         print('---------',self.submodule._modules.items())
#         for name, module in self.submodule._modules.items():
#             if "fc" in name:
#                 x = x.view(x.size(0), -1)
#             print(module)
#             x = module(x)
#             print('name', name)
#             if name in self.extracted_layers:
#                 outputs.append(x)
#         return outputs


# def get_feature():  # 特征可视化
#     # 输入数据
#     img = get_picture(pic_dir, transform) # 输入的图像是【3,256,256】
#     # 插入维度
#     img = img.unsqueeze(0)  # 【1,3,256,256】
#     img = img.to(device)

#     # 特征输出
#     net = L2Net().to(device)
#     #net = L2Net_CBAM().to(device)
#     # net.load_state_dict(torch.load('./model/net_050.pth'))
#     exact_list = ['layer']
#     myexactor = FeatureExtractor(net, exact_list)  # 输出是一个网络
#     x = myexactor(img)

#     # 特征输出可视化
#     for i in range(128):  # 可视化了32通道
#         ax = plt.subplot(12, 12, i + 1)
#         #ax.set_title('Feature {}'.format(i))
#         ax.axis('off')
#         #ax.set_title('new—conv1-image')

#         plt.imshow(x[0].data.cpu().numpy()[0,i,:,:],cmap='jet')
#         print("***************************"+str(i))

#     plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型

#     feature_map_sum = sum(ele for ele in feature_map_combination)
#     plt.imshow(feature_map_sum)
#     plt.savefig("feature_map_sum.png")

# # 训练
# if __name__ == "__main__":
#     #get_picture_rgb(pic_dir)
#     get_feature()
    


# coding: utf-8


