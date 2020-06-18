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

from data import cfg_re50, cfg_mnet
from layers.functions.prior_box import PriorBox
from models.myresnet import resnet50
from models.pfld import PFLDInference
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from torchvision import transforms as T
import torch.nn as nn


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


def process_face_data(im, im_height, im_width, loc, scale, conf, landms):
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

    result_data = dets[:, :5].tolist()

    return result_data


def image_process(im):
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
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).unsqueeze(0)
    im = im.float()
    im = im.cuda()
    return im, im_width, im_height, scale


def mask_recognition(data, img):
    masked = []
    for det in data:
        xmin, ymin, xmax, ymax, conf = det
        w, h = img.size
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        xmax = xmax if xmax < w else w-1
        ymax = ymax if ymax < h else h-1
        im = img.crop((xmin, ymin, xmax, ymax))
        # 3 * 256 * 256
        im = transform1(im)
        im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
        if torch.cuda.is_available():
            im = im.cuda()
        out = mask_net(im)
        out = soft_max(out)
        y = torch.argmax(out, 1)
        y = y.cpu().numpy()[0]
        masked.append(int(y))
    return masked


def get_face_landmarks(data, img):
    landmarks = []
    for det in data:
        xmin, ymin, xmax, ymax, conf = det
        xmin, ymin, xmax, ymax = xmin+0.5, ymin+0.5, xmax+0.5, ymax+0.5
        w, h = img.size
        # xmin = xmin if xmin >= 0 else 0
        # ymin = ymin if ymin >= 0 else 0
        # xmax = xmax if xmax < w else w - 1
        # ymax = ymax if ymax < h else h - 1
        # im = img.crop((xmin, ymin, xmax, ymax))
        crop_w, crop_h = xmax - xmin + 1, ymax - ymin + 1
        scale = int(max([crop_w, crop_h]) * 1.1)
        cx = xmin + crop_w // 2
        cy = ymin + crop_h // 2
        xmin = cx - scale // 2
        xmax = xmin + scale
        ymin = cy - scale // 2
        ymax = ymin + scale

        dx = max(0, -xmin)
        dy = max(0, -ymin)
        xmin = max(0, xmin)
        ymin = max(0, ymin)

        edx = max(0, xmax - w)
        edy = max(0, ymax - h)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        im = img.crop((xmin, ymin, xmax, ymax))
        # im.show()
        im = im.resize((112, 112), Image.ANTIALIAS)
        im = transform2(im).unsqueeze(0).cuda(0)
        _, pre_landmarks = pfld_net(im)
        landmark = pre_landmarks[0].cpu().detach().numpy().reshape(-1, 2) * [scale, scale] + [xmin, ymin]
        landmarks.append(landmark.tolist())
    return landmarks


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
# cfg = cfg_re50
cfg = cfg_mnet
# trained_model = './weights/Resnet50_epoch_95.pth'
retina_trained_model = './weights/mobilenet0.25_epoch_245.pth'
mask_trained_model = './weights/net_21.pth'
pfld_trained_model = './weights/checkpoint_epoch_500.pth.tar'
# net and model
retina_net = RetinaFace(cfg=cfg, phase='test')
mask_net = resnet50(num_classes=2)
pfld_backbone = PFLDInference()
# load pre-trained model
retina_net.load_state_dict(torch.load(retina_trained_model))
mask_net.load_state_dict(torch.load(mask_trained_model))
pfld_backbone.load_state_dict(torch.load(pfld_trained_model)['plfd_backbone'])

retina_net = retina_net.cuda(0)
mask_net = mask_net.cuda(0)
pfld_net = pfld_backbone.cuda(0)
retina_net.eval()
mask_net.eval()
pfld_net.eval()

resize = 1
top_k = 5000
keep_top_k = 750
nms_threshold = 0.5

cudnn.benchmark = True

transform1 = T.Compose([
        T.Resize(size=(256, 256)),
        T.ToTensor(),
        T.Normalize([0.56687369, 0.44000871, 0.39886727], [0.2415682, 0.2131414, 0.19494878])
    ])
transform2 = T.Compose([
    T.ToTensor()
])
soft_max = nn.Softmax()

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
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
            im_pil = Image.open(file)
            im = im_pil.convert('RGB')
            # im = cv2.imread(file)
            im, im_width, im_height, scale = image_process(im)
            tic = time.time()
            loc, conf, landms = retina_net(im)
            # print('net forward time: {:.4f}'.format(time.time() - tic))
            result_data = process_face_data(im, im_height, im_width, loc, scale, conf, landms)
            masked = mask_recognition(result_data, im_pil)
            landmarks = get_face_landmarks(result_data, im_pil)

            [result_data[i].append(masked[i]) for i in range(len(masked))]
            data['success'] = True
            data['prediction'] = result_data
            data['landmarks'] = landmarks

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