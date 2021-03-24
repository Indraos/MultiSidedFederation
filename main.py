from client import Client
from server import Server
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as tt
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
import numpy as np

mnist = False
fashion_mnist = False
cifar = True

n = 3
rounds = 6
batch_size = 100
client_values = [0.1, 0.5, 0.6]
criterion = nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available else "cpu"


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


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if mnist:
    transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
    train_ds = MNIST(root=".", train=True, download=True, transform=transform)
    test_ds = MNIST(root=".", train=False, download=True, transform=transform)

if fashion_mnist:
    transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.2860,), (0.3530,))])
    train_ds = FashionMNIST(root=".", train=True, download=True, transform=transform)
    test_ds = FashionMNIST(root=".", train=False, download=True, transform=transform)

if cifar:
    transform = tt.Compose(
        [tt.ToTensor(), tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_ds = CIFAR10(root=".", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root=".", train=False, download=True, transform=transform)

test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True)

if mnist or fashion_mnist:
    client_dls = utils.iid_clients(train_ds, n, 1000, 10000, batch_size)
    models = [MnistNet() for i in range(n)]
    opt = [Adam(models[i].parameters(), lr=0.01) for i in range(n)]
    clients = [
        Client(
            models[i],
            client_dls[i],
            test_dl,
            criterion,
            device,
            opt[i],
            client_values[i],
        )
        for i in range(n)
    ]

elif cifar:
    client_dls = utils.iid_clients(train_ds, n, 1000, 10000, batch_size)
    models = [CifarNet() for i in range(n)]
    opt = [Adam(models[i].parameters(), lr=0.001) for i in range(n)]
    clients = [
        Client(
            models[i],
            client_dls[i],
            test_dl,
            criterion,
            device,
            opt[i],
            client_values[i],
        )
        for i in range(n)
    ]
server = Server(
    clients, np.zeros_like, np.zeros_like  # no deviation pay, no cross-checking
)

for client in server.clients:
    client.bid()
for i in range(rounds):
    print(f"Round {i}")
    server.run_demand_auction()

server.plot("value", "values.png")
server.plot("utility", "utilities.png")
