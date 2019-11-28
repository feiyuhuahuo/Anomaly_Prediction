#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import cv2
import torch.nn as nn

# aa = cv2.imread('000.jpg') / 255.
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
# loss.backward() ⬃⬆ ⇩↧⬋属于Miscellaneous Symbols and Arrows杂项符号和箭头该分区共有256个，ASCII码为：&#11019,想了解更多内容请来https://www.ziti163.com/uni
# print(net.x2.grad)

import cv2

cv2.namedWindow('aa', cv2.WINDOW_NORMAL)
cv2.resizeWindow('aa', 500, 500)
cv2.moveWindow("aa", 600, 100)

im_gray = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

cv2.imshow('aa', im_color)
cv2.waitKey()