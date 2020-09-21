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

app = FastAPI()

uri = 'mongodb://fang:123456@111.229.203.174:27017/?authSource=facedb&authMechanism=SCRAM-SHA-1'
client = pymongo.MongoClient(uri)
db = client.facedb


def save_file(file):
    contents = await file.read()
    imgput = GridFS(db, collection='face')
    imgput.put(contents, content_type='image/jpeg', filename=file.filename)


@app.post("/uploadFile")
async def uploadFile(file: UploadFile = File(...)):
    # print(type(file))
    return save_file(file)


@app.get("/getFile")
async def getFile(file_id):
    return file_id





