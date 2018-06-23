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
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'dataset'))
sys.path.append(os.path.join(ROOT_DIR,'model'))

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize',type=int,default=32,help='input batch size')
parser.add_argument('--num_points',type=int,default=2500,help='input num points of cloud points')
parser.add_argument('--num_workers',type=int,default=4,help='the num worker of loading data')
parser.add_argument('--num_epoch',type=int,default=50,help='num epoch of training process')
parser.add_argument('--outdir',type=str,default='seg',help='output dir of training')
parser.add_argument('--model',type=str,default='',help='model path')

opt = parser.parse_args()
print(opt)

pointnet = imp.import_module('pointnet')
pointdata = imp.import_module('dataset')

opt.manualSeed = random.randint(1, 10000) 
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_dataset = pointdata.PartDataset(root='../data/shapenetcore_partanno_segmentation_benchmark_v0',npoints=opt.num_points,classification=False,class_choice=['Chair'])
trainLoader = Data.DataLoader(train_dataset,batch_size = opt.batchsize,shuffle=True,num_workers=opt.num_workers)

test_dataset = pointdata.PartDataset(root='../data/shapenetcore_partanno_segmentation_benchmark_v0',npoints=opt.num_points,classification=False,train=False,class_choice=['Chair'])
testLoader = Data.DataLoader(test_dataset,batch_size = opt.batchsize,shuffle=True,num_workers=opt.num_workers)

print(len(train_dataset),len(test_dataset))
num_classes = train_dataset.num_seg_classes
print('num_classes:',num_classes)
try:
	os.makedirs(opt.outdir)
except OSError:
	pass

model = pointnet.PointNetSeg(num_points=opt.num_points,k=num_classes)
if opt.model != '':		
	model.load_state_dict(torch.load(opt.model))
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
model = model.cuda()
model = model.train()
num_batch = int(len(train_dataset)/opt.batchsize)
for epoch in range(opt.num_epoch):
	totalloss = 0
	totalcorrect = 0
	totalnum = 0
	for i,(points,labels) in enumerate(trainLoader):
		points = points.transpose(2,1)
		points = points.cuda();labels = labels.cuda()
		optimizer.zero_grad()
		preds , _ = model(points)
		preds = preds.view(-1,num_classes)
		labels = labels.view(-1,1)[:,0]-1
		loss = F.nll_loss(preds,labels)
		loss.backward()
		optimizer.step()
		index = preds.data.max(1)[1]
		correct = index.eq(labels.data).cpu().sum()
		totalloss += loss.data
		totalnum += len(labels)
		totalcorrect += correct.item()
	print('epoch:{},train loss:{},train accracy:{}'.format(epoch+1,totalloss/totalnum,totalcorrect/totalnum))

	if (epoch+1)% 2 == 0:
		model.eval() 
		totalnum = 0
		totalcorrect = 0
		for i,(points,labels) in enumerate(testLoader):
			points = points.transpose(2,1)
			points = points.cuda();labels = labels.cuda()
			preds , _ = model(points)
			preds = preds.view(-1,num_classes)
			labels = labels.view(-1,1)[:,0]-1
			index = preds.data.max(1)[1]
			correct = index.eq(labels.data).cpu().sum()
			totalnum += len(labels)
			totalcorrect += correct.item()
		print('epoch:{},test accuracy:{}'.format(epoch+1,totalcorrect/totalnum))
		model.train()
	torch.save(model.state_dict(),'%s/seg_model_%d.pth' % (opt.outdir, epoch))

