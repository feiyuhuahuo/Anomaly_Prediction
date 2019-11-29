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
# loss.backward()
# print(net.x2.grad)
import numpy as np
aa = np.array([1,2,3,4])
aa -= min(aa)
print(aa)