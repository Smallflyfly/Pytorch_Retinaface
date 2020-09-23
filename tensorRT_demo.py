# -*- coding: utf-8 -*-
# @Time    : 20-7-4 下午2:42
# @Author  : smallflyfly
# @FileName: tensorRT_demo.py
import time

import cv2

import torch
from torch.backends import cudnn
from torch2trt import torch2trt

from data import cfg_mnet
from models.retinaface import RetinaFace
from utils.net_utils import load_model, image_process, process_face_data

# import torch2trt.converters.cat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
cfg = cfg_mnet
retina_trained_model = "./weights/mobilenet0.25_Final.pth"
use_cpu = False
# cfg = cfg_re50
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, use_cpu)
retina_net.eval()
cudnn.benchmark = True
retina_net = retina_net.to(device)


def main(img_path):
    test_img = cv2.imread(img_path)
    resize = 1
    im, im_width, im_height, scale = image_process(test_img, device)
    print(im.shape)
    model = torch2trt(retina_net, [im], fp16_mode=True, max_workspace_size=100000)
    tic = time.time()
    loc, conf, landms = model(im)
    print('net forward time: {:.4f}'.format(time.time() - tic))
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    print(result_data)


if __name__ == '__main__':
    image = './fang.jpg'
    main(image)