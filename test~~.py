#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
cv2.namedWindow('aa', cv2.WINDOW_NORMAL)
cv2.resizeWindow('aa', 800, 800)
ss = cv2.imread('hen.jpg')
ss = cv2.resize(ss, (100, 100))
cv2.imshow('aa', ss)
cv2.waitKey()