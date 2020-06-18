# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 19:42
# @Author  : Fangpf
# @FileName: client.py
import time

import cv2
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np


REQUEST_URL = "https://xiaomiqiu.ngrok2.xiaomiqiu.cn/upload"


def predict(byte_file):
    im = byte_file
    param = {'file': im}
    res = requests.post(REQUEST_URL, files=param)
    res = res.json()
    predictions = []
    if res['success']:
        predictions = res['prediction']
    return predictions


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        _, jpeg = cv2.imencode('.jpg', small_frame)
        rgb_small_frame = small_frame[:, :, ::-1]
        tic = time.time()
        dets = predict(jpeg.tobytes())
        print('net forward time: {:.4f}'.format(time.time() - tic))
        for det in dets:
            xmin, ymin, xmax, ymax, conf, ismasked = det
            xmin = int(xmin * 4)
            ymin = int(ymin * 4)
            xmax = int(xmax * 4)
            ymax = int(ymax * 4)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            fontStyle = ImageFont.truetype(
                "font/FZY1JW.TTF", 20, encoding="utf-8"
            )
            draw.text((xmin, ymin-20), "佩戴口罩" if ismasked == 1 else "未佩戴口罩", (255, 0, 0), font=fontStyle)
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

        cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
