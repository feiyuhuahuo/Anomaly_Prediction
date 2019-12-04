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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class aa(Dataset):
    def __init__(self):
        self.all_seqs = [[1, 0, 2, 3], [3, 2, 1, 0]]

    def __len__(self):
        return 2

    def __getitem__(self, indice):

        start = self.all_seqs[indice][-1]

        return indice, start

bb = aa()
train_dataloader = DataLoader(dataset=bb, batch_size=1,
                              shuffle=True, num_workers=1, drop_last=True)

for i in range(50):
    for ii, ss in train_dataloader:
        print(ii, ss)

        bb.all_seqs[ii].pop()
        print(bb.all_seqs)

        if len(bb.all_seqs[ii]) == 0:
            print('~~~~~~~')
            bb.all_seqs[ii] = list(range(4))




