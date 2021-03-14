from Client import *
from Server import *
import utils

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

import random
import string


mnist = False
fashion_mnist = False
cifar = False

# Experiments 1: MNIST & Fashion MNIST

n = 3
rounds = 5
epochs = 10

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(), nn.Linear(320, 50), nn.Dropout(), nn.ReLU()
        )
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

criterion = nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available else "cpu"

if mnist:
    transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
    train_ds = MNIST(root=".", train=True, download=True, transform=transform)
    test_ds = MNIST(root=".", train=False, download=True, transform=transform)
	optimizer = Adam(model.parameters(), lr=0.001)

if fashion_mnist:
    transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.2860,), (0.3530,))])
    train_ds = FashionMNIST(root=".", train=True, download=True, transform=transform)
    test_ds = FashionMNIST(root=".", train=False, download=True, transform=transform)
    batch_size = 100
    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_dl = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)


client_dls = utils.iid_clients(train_ds, n)
clients = [Client(MnistNet(), client_dls[i], test_dl, criterion, device) for i in range(n)]
server = Server(MnistNet(), clients, utils.exponential_cutoff)



# Experiments 2: CIFAR



for i in range(rounds):
	for i in range(n):
        clients[i].model = clients[i].train_model()
        _, testacc = clients[i].test(clients[i].model)
        clients[i].val_acc.append(testacc)
    print(f"\nRound {i+1}:")
    print("-------------------------")
    server.run_demand_auction()

for i, client in enumerate(clients):
    client.train_model()
    _, testacc = client.test(client.model)
    client.val_acc.append(testacc)
server.visualize_values(rounds)
server.visualize_utilities(rounds)