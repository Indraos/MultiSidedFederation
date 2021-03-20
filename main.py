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
fashion_mnist = True
cifar = False

n = 3
rounds = 5
epochs = 10
batch_size = 100
client_values = [0.4, 0.5, 0.6]
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


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=4)

if mnist or fashion_mnist:
    client_dls = utils.iid_clients(train_ds, n, 1000, 10000, batch_size)
    clients = [
        Client(
            MnistNet(),
            client_dls[i],
            test_dl,
            criterion,
            device,
            utils.exponential_cutoff,
            client_values[i],
        )
        for i in range(n)
    ]
    server = Server(clients, np.exp, utils.exponential_cutoff)

elif cifar:
    client_dls = utils.iid_clients(train_ds, n, 1000, 10000, batch_size)

    clients = [
        Client(
            CifarNet(),
            client_dls[i],
            test_dl,
            criterion,
            device,
            Adam(),
            utils.transmission_criterion,
            client_values[i],
        )
        for i in range(n)
    ]
    server = Server(clients, np.exp, utils.exponential_cutoff)

for client in clients:
    client.bid()
for _ in range(rounds):
    server.run_demand_auction()

server.plot(rounds, "values.png")
server.plot(rounds, "utilities.png")
