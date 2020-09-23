#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/{DAY}
"""
import io
import time
import uuid
from io import StringIO
from urllib.parse import quote_plus

import bson
import pymongo
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from gridfs import GridFS
import torch.nn as nn
from torch.backends import cudnn

from dao.face import Face
from data import cfg_re50
from database.mongodb import Mongodb
import torchvision.transforms as transform

from database.mysqldb import MySQLDB
from models.retinaface import RetinaFace
from utils.net_utils import image_process, load_model

app = FastAPI()

# uri = 'mongodb://fang:123456@111.229.203.174:27017/?authSource=facedb&authMechanism=SCRAM-SHA-1'
# client = pymongo.MongoClient(uri)
mongodb = Mongodb()
db = mongodb.client.facedb
# print(db)

cfg = cfg_re50
# cfg = cfg_mnet
# trained_model = './weights/Resnet50_epoch_95.pth'
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


@app.post("/uploadFile")
async def uploadFile(file: UploadFile = File(...)):
    contents = await file.read()
    # print(contents)
    gfs = GridFS(db, collection='face')
    file_id = gfs.put(contents, content_type='image/jpeg', filename=file.filename)
    contents = io.BytesIO(contents)
    im_pil = Image.open(contents)
    im_pil = im_pil.resize((256, 256))
    im = im_pil.convert('RGB')
    im, im_width, im_height, scale = image_process(im, device)
    tic = time.time()
    features = retina_net(im).cpu().detach().numpy().reshape((-1,)).tostring()
    # features = ",".join(str(f) for f in features.tolist())
    mysqldb = MySQLDB()
    session = mysqldb.session()
    name = file.filename[:file.filename.index('.')]
    face = session.query(Face).filter(Face.name == name).scaler()
    if face:
        face.feature1 = str(features)
    else:
        face = Face()
        face.user_id = str(uuid.uuid1())
        face.name = name
        # print(features)
        # print(len(features))
        face.feature1 = str(features)
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





