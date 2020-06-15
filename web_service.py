# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 10:41
# @Author  : Fangpf
# @FileName: web_service.py


# You can change this to any folder on your system
import io
import time

import torch
from flask import Flask, request, jsonify
from PIL import Image

from data import cfg_re50
from models.retinaface import RetinaFace

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
net.eval()


@app.route('/', methods=['GET', 'POST'])
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
            im_width, im_height = im.size
            scale = [im_width, im_height, im_width, im_height]
            im -= (104, 117, 123)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im.cuda()
            tic = time.time()
            loc, conf, landms = net(im)

            data['success'] = True
            data['prediction'] = loc
            # print(jsonify(result))
            # return img_stream
                # draw = ImageDraw.Draw(pil_img)
                # print(pil_img)
                # draw.polygon()
            # print(top, right, bottom, left)
            # return detect_faces_in_image(file)

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
    app.run(host='0.0.0.0', port=5001, debug=True)