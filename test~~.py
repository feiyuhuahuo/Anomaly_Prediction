#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import torch
# import cv2
# import torch.nn as nn
#
# aa = cv2.imread('contents/image.png') / 255.
# aa = torch.FloatTensor(aa).permute(2, 0, 1).unsqueeze(0)
#
#
# class ss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = nn.Conv2d(3, 64, 3)
#         self.c2 = nn.Conv2d(64, 32, 3)
#         self.c3 = nn.Conv2d(32, 2, 3)
#
#     def forward(self, x):
#         self.x = x
#         self.x1 = self.c1(self.x)
#         self.x1.retain_grad()
#
#         self.x2_d = self.x1.detach()
#         self.x2_d.requires_grad = True
#
#         self.x2 = self.c2(self.x2_d)
#         self.x2.retain_grad()
#
#         self.x3 = self.c3(self.x2)
#
#         return self.x3
#
#
# net = ss()
# net.train()
#
# out = net(aa)
# loss = torch.sum((out - 1) ** 2)
#
# loss.backward()
# print(net.x1.grad)

import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import cv2

# 使用plt进行画图

ss = Image.open('slark.png')  # 读取图片像素为512X512
fig = plt.figure("Image")  # 图像窗口名称

plt.imshow(ss)

buffer = io.BytesIO()  # 获取输入输出流对象
print(buffer)

fig.canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
data = buffer.getvalue()  # 获取流的值

buffer.write(data)  # 将数据写入buffer
img = np.array(Image.open(buffer))


print("转换的图片array的尺寸为:\n", img.shape)
cv2.imwrite("02.jpg", img)

buffer.close()
