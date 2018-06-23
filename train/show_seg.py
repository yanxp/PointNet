from __future__ import print_function
from __future__ import division
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import importlib as imp
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'dataset'))
sys.path.append(os.path.join(ROOT_DIR,'model'))
sys.path.append(os.path.join(ROOT_DIR,'utils'))
import show3d_ball

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='',help='model path')
parser.add_argument('--index',type=int,default='10',help='data index')
parser.add_argument('--num_points',type=int,default=2500,help='input points num')
opt = parser.parse_args()
print(opt)


pointnet = imp.import_module('pointnet')
pointdata = imp.import_module('dataset')

dataset = pointdata.PartDataset(root='../data/shapenetcore_partanno_segmentation_benchmark_v0',npoints=opt.num_points,classification=False,train=False,class_choice=['Chair'])
print("data index %d/%d" %( opt.index, len(dataset)))

num_classes = dataset.num_seg_classes
print('num_classes:',num_classes)

points, seg = dataset[opt.index]
points_np = points.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg.numpy() - 1, :]

model = pointnet.PointNetSeg(k = num_classes)
model.load_state_dict(torch.load(opt.model))
model.eval()
model.cuda()
points = points.transpose(1,0).contiguous()
points = points.view(1, points.size()[0], points.size()[1])
points = points.cuda()
pred, _ = model(points)
index = pred.data.max(2)[1]
print(index.size())
pred_color = cmap[index.data.cpu().numpy()[0], :]
print(pred_color.shape)
show3d_ball.showpoints(points_np, gt, pred_color)

