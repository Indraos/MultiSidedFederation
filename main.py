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


def set_up_experiment(
    type,
    num_clients=3,
    batch_size=100,
    deviation_pay=np.zeros_like,
    cross_checking=np.zeros_like,
    client_values=[0.8, 0.5, 0.6],
):
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available else "cpu"
    assert type in [
        "mnist",
        "fashion_mnist",
        "cifar",
    ], "Only mnist, fashion_mnist and cifar supported."
    if type == "mnist":
        transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
        train_ds = MNIST(root=".", train=True, download=True, transform=transform)
        test_ds = MNIST(root=".", train=False, download=True, transform=transform)

    if type == "fashion_mnist":
        transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.2860,), (0.3530,))])
        train_ds = FashionMNIST(
            root=".", train=True, download=True, transform=transform
        )
        test_ds = FashionMNIST(
            root=".", train=False, download=True, transform=transform
        )

    if type == "cifar":
        transform = tt.Compose(
            [tt.ToTensor(), tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_ds = CIFAR10(root=".", train=True, download=True, transform=transform)
        test_ds = CIFAR10(root=".", train=False, download=True, transform=transform)

    test_dl = DataLoader(
        test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if type in ["mnist", "fashion_mnist"]:
        client_dls = utils.iid_clients(train_ds, num_clients, 1000, 10000, batch_size)
        models = [MnistNet() for i in range(num_clients)]
        opt = [Adam(models[i].parameters(), lr=0.01) for i in range(num_clients)]
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
            for i in range(num_clients)
        ]

    else:
        client_dls = utils.iid_clients(train_ds, num_clients, 1000, 10000, batch_size)
        models = [CifarNet() for i in range(num_clients)]
        opt = [Adam(models[i].parameters(), lr=0.001) for i in range(num_clients)]
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
            for i in range(num_clients)
        ]
    server = Server(
        clients, np.zeros_like, np.zeros_like  # no deviation pay, no cross-checking
    )
    return server


# Experiment 1: An Auction Simulation
# rounds = 6
# for dataset in ["mnist", "fashion_mnist", "cifar"]:
#     server = set_up_experiment(dataset)
#     for client in server.clients:
#         client.bid()
#     for i in range(rounds):
#         print(f"Round {i}")
#         server.run_demand_auction()
#         for plot_type in ["value", "utility"]:
#             server.plot(plot_type, f"{dataset}_{plot_type}.png")

rounds = 6
deviations = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
deviation_utility = []
for dataset in ["mnist"]:
    server = set_up_experiment(dataset)
    for deviation in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]:
        for client in server.clients:
            client.reset()
        server.clients[0].enter_bid()
        server.clients[2].enter_bid()
        server.clients[1].enter_bid(deviation=deviation)
        for i in range(rounds):
            print(f"Round {i}")
            server.run_demand_auction()
        deviation_utility.append(server.clients[1].utility_history[-1])
print(deviation_utility)
