from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import argparse
import math
import os
import shutil
import stat
import subprocess
import sys
import time
import numpy as np
from skimage import transform
import ctypes
from ctypes import *
import pickle

so = ctypes.cdll.LoadLibrary
librbox = so("../librbox.so")

DecodeAndNMS = librbox.DecodeAndNMS
DecodeAndNMS.argtypes=(POINTER(c_double),POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
DecodeAndNMS.restype=None

NMS = librbox.NMS_ship
NMS.argtypes=(POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
NMS.restype=None

caffe.set_device(0)
caffe.set_mode_gpu()
#caffemodel = 'RBOX_SHIPOPT_RBOX_300x300_SHIPOPT_VGG_new_iter_300000.caffemodel'
#deploy = 'deploy.prototxt'
#net = caffe.Net(deploy,caffemodel,caffe.TEST)
