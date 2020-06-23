# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 15:59
# @Author  : Fangpf
# @FileName: detect.py
import cv2
import torch
from torch.backends import cudnn

from data import cfg_mnet
from models.retinaface import RetinaFace
from utils.net_utils import load_model, process_face_data, image_process

cfg = cfg_mnet
retina_trained_model = "./weights/mobilenet0.25_Final.pth"
# cfg = cfg_re50
use_cpu = False
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, use_cpu)
retina_net.eval()
print('Finished loading model!')
cudnn.benchmark = True
device = torch.device("cpu" if use_cpu else "cuda")


def retina_detect(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


if __name__ == '__main__':
    url = "rtsp://admin:fang2831016@172.27.12.188:554/stream1"
    video_capture = cv2.VideoCapture(url)
    while True:
        _, frame = video_capture.read()
        face_result = retina_net(frame)
        for det in face_result:
            xmin, ymin, xmax, ymax, conf = det
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)

        cv2.imshow('face detection', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


