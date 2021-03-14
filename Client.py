import string
import random
import torch
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score


class Client:
    client_num = 0  # index for naming purposes
    clients = []
	bids = []

    def __init__(self, architecture, train_dl, test_dl, criterion, device, bid):
        # n-people and their bids
        # ['A', 'B', .... 'Z'] -> list of alphabets for visualization purpose
        self.person = list(string.ascii_uppercase)[Client.client_num]
        Client.clients.append(self)
		Client.bids.append(bid)
        Client.client_num += 1

        self.receivers = []
        self.architecture = architecture
        self.device = device
        self.test_dl = test_dl
        self.train_dl = train_dl
        self.criterion = criterion

		self.pay = 0
        self.bid = 0

        self.received_models = []
        self.evaluated_models = []
        self.eval_acc = []
		self.median_score = 0
		self.median_aggregated_score = 0
		self.score = 0
		self.aggregated_score = 0

    def send_model(self):
        """
        Send model to each client in self.receivers.
        """
        for receiver in self.receivers:
            receiver.received_models.append(self.architecture)
        self.received_models = []
        self.receivers = []

    def aggregate_and_send(self):
        """
        Aggregate all received models and append models to received models
        """
        # Average all models
        new_state_dict = {k: 0 for k in new_model.state_dict()}
        for model in self.received_models:
            s_dict = model.state_dict()
            for k, v in s_dict.items():
                new_state_dict[k] += v
        new_state_dict = {
            k: v / len(self.received_models) for k, v in new_state_dict.items()
        }
        new_model = self.architecture
        new_model.load_state_dict(new_state_dict)
        return new_model

    def test(self, input_model):	
        model = input_model
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
            print(f"\nAverage Val Loss: {testl:.4f}, Val Accuracy: {testacc:.4f}\n")
            return testl, testacc

    def evaluate(self):
        """
        Evaluate all models in received_models and append to evaluated_models
        """
        self.evaluated_models = []
        self.eval_acc = []

        # Evaluating own model
        _, testacc = self.test(self.architecture)
        self.eval_acc.append(testacc)
        self.evaluated_models.append(self.architecture)

        # Evaluating received models (single)
        for i in range(len(self.received_models)):
            # print("Evaluating " + self.received_models[i].person + "...")
            _, testacc = self.test(self.received_models[i])
            self.eval_acc.append(testacc)
            self.evaluated_models.append(self.received_models[i])

        # Evaluating aggregate of received models

        if len(self.received_models) > 1:
            print("Aggregated model")
            agg_model = self.aggregate()
            _, testacc = self.test(agg_model)
            self.eval_acc.append(testacc)
            self.evaluated_models.append(agg_model)

    def pay_amt(self, amount):
        """
        Change the amount to bid by a value
        """
        self.pay += amount

    def train_model(self):
        """
        Train one round using newly arrived data.
        """
        model = self.architecture
        print(f"Training client {self.person}...")
        model.to(self.device)
        model.train()
        optimizer = Adam(model.parameters(), lr=0.001)
        
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