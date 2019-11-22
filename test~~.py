#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch

x = torch.Tensor([2., 3.])
x.requires_grad = True

y = 2 * x ** 2
y.retain_grad()

y2 = y.detach()
z = 3 * y2 + 4
z.requires_grad =True
z.retain_grad()

l = torch.sum(z ** 2)
l.retain_grad()
l.backward()
print(y2.grad)
