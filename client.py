# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 19:42
# @Author  : Fangpf
# @FileName: client.py
import cv2
import requests


REQUEST_URL = "https://xiaomiqiu.ngrok2.xiaomiqiu.cn/upload"


def predict(byte_file, frame):
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
        dets = predict(jpeg.tobytes(), frame)
        for det in dets:
            xmin, ymin, xmax, ymax, conf = det
            xmin *= 4
            ymin *= 4
            xmax *= 4
            ymax *= 4
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
        cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
