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

from lib.opts import opts
from tools.kitti_eval.pytorch2caffe import Pytorch2Caffe
from lib.datasets.dataset_factory import dataset_factory
from lib.models.model import create_model, load_model

def prefetch_test(opt):
  
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  #model = load_model(self.model, opt.load_model)
  model.eval()
  p2c = Pytorch2Caffe(model, '/home/lijf/MT', 'parking', [3, 512, 512])
  p2c.start()

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)