#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/{DAY}
"""
from io import StringIO
from urllib.parse import quote_plus

import bson
import pymongo
from fastapi import FastAPI, UploadFile, File
from gridfs import GridFS

from database.mongodb import Mongodb

app = FastAPI()

# uri = 'mongodb://fang:123456@111.229.203.174:27017/?authSource=facedb&authMechanism=SCRAM-SHA-1'
# client = pymongo.MongoClient(uri)
mongodb = Mongodb()
db = mongodb.client.facedb


def save_file(file):
    contents = file.read()
    gfs = GridFS(db, collection='face')
    gfs.put(contents, content_type='image/jpeg', filename=file.filename)


@app.post("/uploadFile")
async def uploadFile(file: UploadFile = File(...)):
    # print(type(file))
    return save_file(file)


@app.get("/getFile")
async def getFile(file_id):
    gfs = GridFS(db, collection='face')
    image_file = gfs.find_one(file_id)
    print(image_file)
    return file_id





