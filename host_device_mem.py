#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2021/07/12
"""

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()