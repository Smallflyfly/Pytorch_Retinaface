#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/24
"""
from face_center.tensorrt.centerface import CenterFace

centerface = CenterFace()


def face_detection(im):
    h, w = im.shape[:2]
    dets, lms = centerface(im, h, w, threshold=0.5)
    # 做人脸录入 只允许一个人
    if len(dets) == 0:
        return [], []
    det = dets[0]
    boxes, score = det[:4], det[4]
    return boxes, score
