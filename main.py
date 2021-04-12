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
import matplotlib.pyplot as plt

n = 3
rounds = 2
batch_size = 100
client_values = [0.1, 0.5, 0.6]
datasets = ["mnist", "fashion_mnist", "cifar"]
deviations = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
fixed_split = [0.5, 0.4, 0.1]
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


def set_up(dataset):
    if dataset == "mnist":
        transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
        train_ds = MNIST(root=".", train=True, download=True, transform=transform)
        test_ds = MNIST(root=".", train=False, download=True, transform=transform)

    if dataset == "fashion_mnist":
        transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.2860,), (0.3530,))])
        train_ds = FashionMNIST(
            root=".", train=True, download=True, transform=transform
        )
        test_ds = FashionMNIST(
            root=".", train=False, download=True, transform=transform
        )

    if dataset == "cifar":
        transform = tt.Compose(
            [tt.ToTensor(), tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_ds = CIFAR10(root=".", train=True, download=True, transform=transform)
        test_ds = CIFAR10(root=".", train=False, download=True, transform=transform)

    test_dl = DataLoader(
        test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if dataset in ["mnist", "fashion_mnist"]:
        client_dls = utils.iid_clients(
            train_ds, n, 1000, 10000, batch_size, fixed_split
        )
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

    elif dataset == "cifar":
        client_dls = utils.iid_clients(
            train_ds, n, 1000, 10000, batch_size, fixed_split
        )
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
    return server, clients


experiment_1 = False
experiment_2 = True
# Experiment 1
if experiment_1:
    for dataset in datasets:
        server, clients = set_up(dataset)
        for client in server.clients:
            client.bid()
        for i in range(rounds):
            print(f"Round {i}")
            server.run_demand_auction()
        server.plot("value", f"{dataset}_values.png")

# Experiment 2
if experiment_2:
    for dataset in datasets:
        utilities = []
        for deviation in deviations:
            server, clients = set_up(dataset)
            clients[0].bid()
            clients[1].bid(deviation)
            clients[2].bid()
            for i in range(rounds):
                print(f"Round {i}")
                server.run_demand_auction()
            utilities.append(
                clients[1].bid * sum(clients[1].allocation_history)
                - sum(clients[1].payment_history)
            )
        plt.plot(deviations, utilities)
        plt.ylabel("Objective")
        plt.xlabel("Deviation")
        plt.grid(True)
        plt.savefig(f"{dataset}_deviations.png", bbox_inches="tight")
        plt.clf()
