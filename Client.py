import string
import random
import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score


class Client:
    client_num = 0  # index for naming purposes
    clients = []
    bids = []

    def __init__(self, architecture, train_dl, test_dl, criterion, device, optimizer, bid):
        Client.clients.append(self)
        self.client_num = Client.client_num
        Client.bids.append(bid)
        Client.client_num += 1

        self.architecture = architecture
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.bid = bid

        self.receivers = []
		self.pay = 0

        self.to_send = []
        self.to_evaluate = {}
        self.evaluations_single = {}
        self.evaluations_aggregated = {}
        self.median_accuracy = 0

	def __str__(self):
		return list(string.ascii_uppercase)[self.client_num]

    def send(self,aggregate,identity):
        """
        Send model to each client in self.receivers and empty own received models.
        """
        for receiver in self.receivers:
            receiver.to_evaluate[(self.client_num, aggregate)].extend(to_send)
        self.to_send = []

    def aggregate(self):
        """
        Aggregate all received models and append models to received models
        """
        new_model = self.architecture
        new_state_dict = {k: 0 for k in new_model.state_dict()}
        for model in self.received_models:
            s_dict = model.state_dict()
            for k, v in s_dict.items():
                new_state_dict[k] += v
        new_state_dict = {
            k: v / len(self.received_models) for k, v in new_state_dict.items()
        }
        new_model.load_state_dict(new_state_dict)
        return new_model

    def test(self, model):	
        with torch.no_grad():
            model.to(self.device)
            model.eval()
            batch_loss, batch_acc = [], []
            for features, labels in self.test_dl:
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                logits = model(features)
                loss = self.criterion(logits, labels)
                batch_loss.append(loss.cpu())
                pred = torch.argmax(logits, dim=1)
                batch_acc.append(accuracy_score(labels.cpu(), pred.cpu()))
            model.cpu()
            testl = sum(batch_loss) / len(batch_loss)
            testacc = round(sum(batch_acc) / len(batch_acc), 4)
            return testl, testacc

    def evaluate(self, identity):
        for (source, aggregate), model in self.to_evaluate.iteritems():
            _, testacc = self.test(model)
            if aggregate:
                Client.clients[source].evaluations_single[identity] = test_acc
            else:
                Client.clients[source].evaluations_aggregated[identity] = test_acc
        self.to_evaluate = {}

    def train(self):
        model = self.architecture
        print(f"Training client {self.person}...")
        model.to(self.device)
        model.train()
        
		batch_loss, batch_acc = [], []
        for features, labels in self.train_dl:
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.cpu())
            pred = torch.argmax(logits, dim=1)
            batch_acc.append(accuracy_score(labels.cpu(), pred.cpu()))

        train_loss = sum(batch_loss) / len(batch_loss)
        train_acc = round(sum(batch_acc) / len(batch_acc), 4)
        print(f"\nAverage Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
        self.architecture = model