"""
Instantiate MNIST/Fashion MNIST/CIFAR dataset here.
"""

from Client import *
from Server import *

import numpy as np 
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import MNIST, FashionMNIST, ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader, Subset

from sklearn.metrics import accuracy_score

import os
import time
import pickle
import PIL

transform = tt.Compose([tt.ToTensor(),
					tt.Normalize((0.1307,), (0.3081,))])

train_ds = MNIST(root='.', train=True, download=True, transform=transform)
test_ds = MNIST(root='.', train=False, download=True, transform=transform)

# transform = tt.Compose([tt.ToTensor(),
#                     tt.Normalize((0.2860,), (0.3530,))])

# train_ds = FashionMNIST(root='.', train=True, download=True, transform=transform)
# test_ds = FashionMNIST(root='.', train=False, download=True, transform=transform)

batch_size=100

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers = 4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size, num_workers = 4, pin_memory=True)

class MnistNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 5),
								   nn.MaxPool2d(2),
								   nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5),
								   nn.Dropout2d(),
								   nn.MaxPool2d(2),
								   nn.ReLU())
		self.fc1 = nn.Sequential(nn.Flatten(),
								 nn.Linear(320, 50),
								 nn.Dropout(),
								 nn.ReLU())
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x



epochs = 10
model = MnistNet()
optimizer = Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available else 'cpu'



#Splitting dataset in (length)-unequal parts (lower = lower limit, upper = upper limit for every split)
def num_pieces(num,length, lower, upper):
	ot = list(range(1,length+1))[::-1]
	all_list = []
	for i in range(length-1):
		n = random.randint(lower, min(upper,num-ot[i]))
		all_list.append(n)
		num -= n
	all_list.append(num) 
	return all_list


def iid_clients(train_ds, n):
	# size = len(train_ds)//n
	# last_size = size + len(train_ds)%n
	# print('Dataset split: ',[size]*(n-1) + [last_size])
	# client_ds = random_split(train_ds, [size]*(n-1) + [last_size])

	split = num_pieces(len(train_ds), n, 1000, 10000)
	print('Dataset split: ',split)
	client_ds = random_split(train_ds, split)
	
	client_dls = [DataLoader(ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)for ds in client_ds]
	# client_models = [MnistNet() for _ in range(n)]
	# client_optimizers = [Adam(model.parameters(), 0.001) for model in client_models]
	return client_dls


#Transmission function which takes bid and threshold as argument
def transmission_criterion(bid, threshold):
	return (1-np.exp(-bid)) > threshold

#n -> number of clients
n = 3
clients = []

client_dls = iid_clients(train_ds, n)

for i in range(n):
	clients.append(DemandClient(client_dls[i], test_dl, optimizer, criterion, model, device))
	clients[i]

client_dls, client_models, client_optimizers = iid_clients(train_ds, n)

Server_main = Server(model, supply_clients, demand_clients, leakage, punishment, transmission_criterion)

rounds = 5

for i in range(rounds):
	Server_main.run_demand_auction()


Server_main.visualize_values(rounds)




  

