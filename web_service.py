# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 10:41
# @Author  : Fangpf
# @FileName: web_service.py


# You can change this to any folder on your system
import io
import time

import cv2
import torch
from flask import Flask, request, jsonify
from PIL import Image
from torch.backends import cudnn
import numpy as np

from data import cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


cfg = cfg_re50
trained_model = './weights/Resnet50_epoch_95.pth'
# net and model
net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, trained_model)
net.cuda()
cudnn.benchmark = True
net.eval()
resize = 1
top_k = 5000
keep_top_k = 750
nms_threshold = 0.5


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    global net
    data = {'success': False}
    if request.method == 'POST':
        # Check if a valid image file was uploaded
        if 'file' not in request.files:
            return jsonify(data)
        if not request.files.get('file'):
            return jsonify(data)
        file = request.files['file'].read()

        # if file and allowed_file(file.filename):
        if file:
            # The image file seems valid! Detect faces and return the result.
            file = io.BytesIO(file)
            im = Image.open(file)
            im = im.convert('RGB')
            # im = cv2.imread(file)
            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            # covert BGR to RGB
            im = im[:, :, ::-1]
            im = np.array(im).astype(int)
            im_width, im_height = im.shape[1], im.shape[0]
            scale = [im_width, im_height, im_width, im_height]
            scale = torch.from_numpy(np.array(scale))
            scale = scale.float()
            scale = scale.cuda()
            im -= (104, 117, 123)
            im = im.transpose(2, 0, 1)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im.float()
            im = im.cuda()
            tic = time.time()
            loc, conf, landms = net(im)
            # print('net forward time: {:.4f}'.format(time.time() - tic))
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.cuda()
            priors_data = priors.data
            boxes = decode(loc.data.squeeze(0), priors_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).cpu().detach().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), priors_data, cfg['variance'])
            scale_landm = torch.from_numpy(np.array([
                im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                im.shape[3], im.shape[2]
            ]))
            scale_landm = scale_landm.float()
            scale_landm = scale_landm.cuda()
            landms = landms * scale_landm / resize
            landms = landms.cpu().numpy()

            # ignore low score
            inds = np.where(scores > 0.6)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = np.argsort(-scores)[:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do nms
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(float, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K fater NMS
            dets = dets[:keep_top_k, :]
            landms = landms[:keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # print(dets[:, :5])

            result_data = dets[:, :5].tolist()

            data['success'] = True
            data['prediction'] = result_data

    # If no valid image file was uploaded, show the file upload form:
    # print(data)
    return jsonify(data)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)