#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2021/07/09
"""
import time

import pycuda.autoinit
import pycuda.driver as cuda
import os
import tensorrt as trt
import cv2
import numpy as np

from data import cfg_mnet
from host_device_mem import HostDeviceMem

import torch

from utils.net_utils import process_face_data, process_face_data_cpu
from math import ceil

TRT_LOGGER = trt.Logger()
BATCH_SIZE = 1
EXPLICIT_BATCH = 1

INPUT_W = 640
INPUT_H = 640

CFG = cfg_mnet

cuda.init()
device = cuda.Device(0)
ctx = device.make_context()


def onnx2trt(onnx_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = BATCH_SIZE
        builder.fp16_mode = True
        if not os.path.exists(onnx_file):
            raise FileNotFoundError("Onnx file {} not found".format(onnx_file))
        with open(onnx_file, 'rb') as onnx:
            parser.parse(onnx.read())

        network.get_input(0).shape = [1, 3, 640, 640]
        engine = builder.build_engine(network, config)
        with open(engine_file, 'wb') as f:
            f.write(engine.serialize())

    return engine


def read_engine(engine_file):
    if not os.path.exists(engine_file):
        raise FileNotFoundError("engine file {} not found!".format(engine_file))
    with open(engine_file, 'rb') as f:
        run_time = trt.Runtime(TRT_LOGGER)
        engine = run_time.deserialize_cuda_engine(f.read())
        return engine


def get_engine(onnx_file, engine_file):
    if not os.path.exists(engine_file):
        # onnx 2 trt
        engine = onnx2trt(onnx_file)
    else:
        engine = read_engine(engine_file)
    return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # allocate
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def prepare(im):
    # BGR 2 RGB
    im = im[:, :, ::-1]
    w, h = im.shape[1], im.shape[0]
    if w > h:
        r = 640 / w
    else:
        r = 640 / h
    im_new = np.zeros((INPUT_H, INPUT_W, 3)).astype("float32")
    w1, h1 = int(r * w), int(r * h)
    pw = int((INPUT_W - w1) // 2)
    ph = int((INPUT_H - h1) // 2)
    im = cv2.resize(im, (0, 0), fx=r, fy=r)
    im = np.array(im).astype("float32")
    im -= (104, 117, 123)
    im_new[ph:ph+h1, pw:pw+w1, :] = im[:, :, :]
    im_new = im_new.transpose((2, 0, 1))
    im_new = im_new.reshape((1, 3, INPUT_H, INPUT_W))
    return im_new, r, pw, ph


def doInference(context, inputs, outputs, bindings, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # do inference
    context.execute_async_v2(bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


if __name__ == '__main__':
    onnx_file = "retinaFace.onnx"
    engine_file = "retinaFace.trt"
    engine = get_engine(onnx_file, engine_file)
    im = cv2.imread('./curve/test.jpg')
    im1, r, pw, ph = prepare(im)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = im1.reshape(-1)
    for i in range(100):
        t1 = time.time()
        ctx.push()
        output = doInference(context, inputs, outputs, bindings, stream)
        ctx.pop()
        # three output prob bbox landmark
        probs = output[2].reshape(-1, 2)
        bbox = output[0].reshape(-1, 4)
        landmarks = output[1].reshape(-1, 10)
        scale = [INPUT_H, INPUT_W, INPUT_H, INPUT_W]
        scale = np.array(scale).astype('float32')
        result_data = process_face_data_cpu(CFG, im1, 640, 640, bbox, scale, probs, landmarks, 1)
        for det in result_data:
            xmin, ymin, xmax, ymax, conf = det
            xmin, ymin, xmax, ymax = (xmin - pw) / r, (ymin - ph) / r, (xmax - pw) / r, (ymax - ph) / r
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
        print(time.time() - t1)
    ctx.pop()

    cv2.imshow("im", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()