#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/{DAY}
"""
import io
import time
import uuid

import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from gridfs import GridFS
from torch.backends import cudnn

from dao.face import Face
from data import cfg_re50
from database.mongodb import Mongodb

from database.mysqldb import MySQLDB
from face_center.python.face_detection import face_detection
from models.retinaface import RetinaFace
from utils.net_utils import image_process, load_model
import numpy as np

app = FastAPI()
mongodb = Mongodb()
db = mongodb.client.facedb

cfg = cfg_re50
retina_trained_model = './weights/Resnet50_Final.pth'
# mask_trained_model = './weights/net_21.pth'
# pfld_trained_model = './weights/checkpoint_epoch_500.pth.tar'
# net and model
retina_net = RetinaFace(cfg=cfg, phase='test')
# mask_net = resnet50(num_classes=2)
# pfld_backbone = PFLDInference()
# load pre-trained model
# retina_net.load_state_dict(torch.load(retina_trained_model))
retina_net = load_model(retina_net, retina_trained_model, False)
# mask_net.load_state_dict(torch.load(mask_trained_model))
# pfld_backbone.load_state_dict(torch.load(pfld_trained_model)['plfd_backbone'])

retina_net = retina_net.cuda(0)
# mask_net = mask_net.cuda(0)
# pfld_net = pfld_backbone.cuda(0)
retina_net.eval()
# mask_net.eval()
# pfld_net.eval()

resize = 1
top_k = 5000
keep_top_k = 750
nms_threshold = 0.5

cudnn.benchmark = True

# transform1 = transform.Compose([
#         transform.Resize(size=(256, 256)),
#         transform.ToTensor(),
#         transform.Normalize([0.56687369, 0.44000871, 0.39886727], [0.2415682, 0.2131414, 0.19494878])
#     ])
# transform2 = transform.Compose([
#     transform.ToTensor()
# ])
# soft_max = nn.Softmax()
device = torch.device("cuda")


@app.post("/face/upload")
async def uploadFile(file: UploadFile = File(...)):
    contents = await file.read()
    gfs = GridFS(db, collection='face')
    file_id = gfs.put(contents, content_type='image/jpeg', filename=file.filename)
    im_pil, cv_im = init_image(contents)
    boxes, score = face_detection(cv_im)
    if len(boxes) == 0:
        return {"code": 400, "success": False, "message": "未检测出人脸，请重新上传"}
    # print(boxes, score)
    im_pil = im_pil.crop([boxes[0], boxes[1], boxes[2], boxes[3]])
    features = generate_feature(im_pil)
    mysqldb = MySQLDB()
    session = mysqldb.session()
    name = file.filename[:file.filename.index('.')]
    face = session.query(Face).filter(Face.name == name).scalar()
    if face:
        face.feature1 = features
        # print(type(features))
        array = string2array_in(features)
        # print(array.shape)
    else:
        face = Face()
        face.user_id = str(uuid.uuid1())
        face.name = name
        # print(features)
        # print(len(features))
        face.feature1 = features
        face.image_url = str(file_id)
        session.add(face)
    session.commit()
    session.close()
    # print(len(features.tostring()))
    # print('net forward time: {:.4f}'.format(time.time() - tic))
    # print(str(file_id))

    return {"code": 200, "success": True, "file_id": str(file_id)}


@app.get("/getFile")
async def getFile(file_id):
    gfs = GridFS(db, collection='face')
    image_file = gfs.find_one(file_id)
    print(image_file)
    return file_id


@app.post("/face/match")
async def faceMatch(file: UploadFile = File(...)):
    contents = await file.read()
    im_pil, cv_im = init_image(contents)
    boxes, score = face_detection(cv_im)
    if len(boxes) == 0:
        return {"code": 400, "success": False, "message": "未检测出人脸，请重新上传"}
    im_pil = im_pil.crop([boxes[0], boxes[1], boxes[2], boxes[3]])
    feature_in = generate_feature(im_pil)
    array_in = string2array_in(feature_in).reshape((64, 64))
    # print(array_in.shape)
    torch_in_feature = torch.from_numpy(array_in)
    # print(torch_in_feature.shape)
    mysqldb = MySQLDB()
    session = mysqldb.session()
    faces = session.query(Face).all()
    max_similarity = 0.0
    for face in faces:
        feature_db = face.feature1
        array_db = string2array_db(feature_db)
        # print(array_db)
    # fang[-1]


def generate_feature(im):
    im = im.resize((256, 256))
    im = im.convert('RGB')
    im, im_width, im_height, scale = image_process(im, device)
    tic = time.time()
    _, features = retina_net(im)
    features = features.cpu().detach().numpy().reshape((-1,))
    # print(features)
    features = features.tostring()
    print(time.time() - tic)
    return features


def init_image(contents):
    content = io.BytesIO(contents)
    im_pil = Image.open(content)
    cv_im = cv2.cvtColor(np.asarray(im_pil), cv2.COLOR_RGB2BGR)
    return im_pil, cv_im


def string2array_in(feature):
    array = np.fromstring(feature, dtype=np.float32)
    return array


def string2array_db(feature):
    array = np.fromstring(feature, dtype=np.float32)
    return array
