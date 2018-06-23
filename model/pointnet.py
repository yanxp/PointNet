
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

class TransLayer(nn.Module):
	"""docstring for TransLayer"""
	def __init__(self, num_points=2500):
		super(TransLayer, self).__init__()
		self.num_points = num_points
		self.layer1 = nn.Sequential(nn.Conv1d(3,64,1),
					nn.BatchNorm1d(64),
					nn.ReLU(),
					nn.Conv1d(64,128,1),
					nn.BatchNorm1d(128),
					nn.ReLU(),
					nn.Conv1d(128,1024,1),
					nn.BatchNorm1d(1024),
					nn.ReLU(),
					nn.MaxPool1d(num_points)
					)
		self.layer2 = nn.Sequential(nn.Linear(1024,512),
						nn.BatchNorm1d(512),
						nn.ReLU(),
						nn.Linear(512,256),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						nn.Linear(256,9))
	def forward(self,x):
		batchsize = x.size()[0]
		x = self.layer1(x)
		x = x.view(-1,1024)
		x = self.layer2(x)
		iden = torch.Tensor([1,0,0,0,1,0,0,0,1]).view(1,9).repeat(batchsize,1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x+iden
		x = x.view(-1,3,3)
		return x
class PointNetfeat(nn.Module):
	"""docstring for PointNetfeat"""
	def __init__(self, num_points=2500,global_feat=True):
		super(PointNetfeat, self).__init__()
		self.stn = TransLayer(num_points=num_points)
		self.conv1 = nn.Conv1d(3,64,1)
		self.conv2 = nn.Conv1d(64,128,1)
		self.conv3 = nn.Conv1d(128,1024,1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.mp = nn.MaxPool1d(num_points)
		self.num_points = num_points
		self.global_feat = global_feat
	def forward(self,x):
		batchsize = x.size()[0]
		trans = self.stn(x) # b,3,3
		x = x.transpose(2,1) # b,num_points,3
		x = torch.bmm(x,trans) # b num_points,3
		x = x.transpose(2,1) # b 3 num_points
		x = F.relu(self.bn1(self.conv1(x)))
		pointfeat = x # b 64 num_ponits
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x = self.mp(x) # b 1024
		x = x.view(-1,1024)
		if self.global_feat:
			return x,trans # b 1024
		else:
			x = x.view(-1,1024,1).repeat(1,1,self.num_points) # b 1024 num_points
			return torch.cat([x,pointfeat],dim=1),trans # b 1088 num_points
class PointNetCls(nn.Module):
	"""docstring for PointNetCls"""
	def __init__(self, num_points=2500,k=2):
		super(PointNetCls, self).__init__()
		self.num_points = num_points
		self.feat = PointNetfeat(num_points,global_feat=True)
		self.layer = nn.Sequential(nn.Linear(1024,512),
					nn.BatchNorm1d(512),
					nn.ReLU(),
					nn.Linear(512,256),
					nn.BatchNorm1d(256),
					nn.ReLU(),
					nn.Linear(256,k))
	def forward(self,x):
		x , trans = self.feat(x) # b 1024
		x = self.layer(x) # b k
		return F.log_softmax(x,dim=-1),trans
class PointNetSeg(nn.Module):
	"""docstring for PointNetSeg"""
	def __init__(self, num_points=2500,k=2):
		super(PointNetSeg, self).__init__()
		self.num_points = num_points
		self.k = k
		self.feat = PointNetfeat(num_points,global_feat=False)
		self.layer = nn.Sequential(nn.Conv1d(1088,512,1),
					nn.BatchNorm1d(512),
					nn.ReLU(),
					nn.Conv1d(512,256,1),
					nn.BatchNorm1d(256),
					nn.ReLU(),
					nn.Conv1d(256,128,1),
					nn.BatchNorm1d(128),
					nn.Conv1d(128,self.k,1))
	def forward(self,x):
		batchsize = x.size()[0]
		x,trans = self.feat(x) # b 1088 num_points
		x = self.layer(x) # b k num_points
		# print(x.size())
		x = x.transpose(2,1).contiguous() # b num_points k
		x = F.log_softmax(x.view(-1,self.k),dim=-1) # b*num_points k
		x = x.view(batchsize,self.num_points,self.k) # b num_points k
		return x,trans
			
		
if __name__ == '__main__':
	sim_data = torch.empty((32,3,2500)).random_(1)
	trans = TransLayer()
	out = trans(sim_data)
	print('stn3d:',out.size())
	pointfeat = PointNetfeat(global_feat=False)
	out,_ = pointfeat(sim_data)
	print('pointfeat:',out.size())
	pointfeat = PointNetfeat(global_feat=True)
	out, _ = pointfeat(sim_data)
	print('point feat', out.size())
	cls = PointNetCls(k = 5)
	out, _ = cls(sim_data)
	print('class', out.size())
	seg = PointNetSeg(k = 3)
	out, _ = seg(sim_data)
	print('seg', out.size())