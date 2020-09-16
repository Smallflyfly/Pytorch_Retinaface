# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 22:33
# @Author  : Fangpf
# @FileName: camera.py
import cv2


def demo():
    # url = 'rtsp://admin:fang2831016@192.168.1.109:554/stream1'
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()