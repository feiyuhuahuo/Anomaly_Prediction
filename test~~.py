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

# import matplotlib.pyplot as plt
# import time
# from math import *
#
# plt.figure(1)
#
# t = []
# m = []
#
# for i in range(2000):
#     plt.clf()  # 清空画布上的所有内容
#
#     t_now = i * 0.1
#
#     t.append(t_now)  # 模拟数据增量流入，保存历史数据
#     m.append(sin(t_now))  # 模拟数据增量流入，保存历史数据
#
#     aa = time.time()
#     if i > 1:
#         print(aa - temp)
#     temp = aa
#
#     plt.plot(t, m, '-r')
#     plt.pause(0.0001)

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def beta_pdf(x, a, b):
    return (x ** (a - 1) * (1 - x) ** (b - 1) * math.gamma(a + b)
            / (math.gamma(a) * math.gamma(b)))


class UpdateDist(object):
    def __init__(self, ax):
        self.success = 0
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)

    def __call__(self, i):
        print(i)
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)

        self.line.set_data(self.x, y)

        return self.line,


# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 15)
ax.grid(True)

ud = UpdateDist(ax)

anim = FuncAnimation(fig, ud, frames=np.arange(100), interval=50, blit=True)

plt.show()


