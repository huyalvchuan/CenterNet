from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from opts import opts
from tools.kitti_eval import pytorch2caffe
from models.model import create_model, load_model

def prefetch_test(opt):
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  model = load_model(self.model, opt.load_model)
  model.eval()
  p2c = pytorch2caffe(model, '/home/lijf/MT', 'parking', [3, 512, 512])

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)