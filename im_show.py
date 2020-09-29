#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/28
"""
import os

import cv2
import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torchvision import transforms

from models.face_detection import face_detection
from utils.net_utils import cosine_similarity

transform = transforms.Compose(
    [transforms.ToTensor()]
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

if __name__ == '__main__':
    im = cv2.imread('jay-2.jpg')
    dets = face_detection(im)
    im_pil = Image.open('jay-2.jpg')
    w, h = im_pil.size
    print(w, h)
    fang[-1]

    det = dets[0]
    boxes, score = det[:4], det[4]
    im_pil_crop = im_pil.crop([boxes[0], boxes[1], boxes[2], boxes[3]])
    im_pil_crop = im_pil_crop.resize((128, 128))
    # im_pil_crop.show()
    im_pil_crop_tensor = transform(im_pil_crop)
    img_embedding = resnet(im_pil_crop_tensor.unsqueeze(0))
    features = img_embedding[0].cpu().detach().numpy().tolist()

    max_cos_similarity = 0.0
    min_distance = 99999.0
    cos_image = None
    distance_image = None
    images = os.listdir('./images/')
    for image in images:
        im1 = cv2.imread('./images/' + image)
        dets1 = face_detection(im1)
        im_pil1 = Image.open('./images/' + image)
        det1 = dets1[0]
        boxes1, score1 = det1[:4], det1[4]
        im_pil_crop1 = im_pil1.crop([boxes1[0], boxes1[1], boxes1[2], boxes1[3]])
        im_pil_crop1 = im_pil_crop1.resize((128, 128))
        # im_pil_crop1.show()
        im_pil_crop_tensor1 = transform(im_pil_crop1)
        img_embedding1 = resnet(im_pil_crop_tensor1.unsqueeze(0))
        features1 = img_embedding1[0].cpu().detach().numpy().tolist()
        similarity = cosine_similarity(features1, features)
        similarity_torch = torch.pairwise_distance(img_embedding1, img_embedding)
        if similarity > max_cos_similarity:
            max_cos_similarity = similarity
            cos_image = image
        if min_distance > similarity_torch.cpu().detach().numpy()[0]:
            min_distance = similarity_torch.cpu().detach().numpy()[0]
            distance_image = image



    print(max_cos_similarity)
    print(cos_image)
    print(min_distance)
    print(distance_image)

