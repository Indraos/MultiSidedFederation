import string
import torch
from sklearn.metrics import accuracy_score


class Client:
    client_count = 0

    def __init__(
        self, architecture, train_dl, test_dl, criterion, device, optimizer, value
    ):

        self.architecture = architecture

        self.train_dl = train_dl
        self.test_dl = test_dl
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.value = value

        self.client_num = Client.client_count
        Client.client_count += 1

        self.model = self.architecture.state_dict()
        self.receivers = []
        self.pay = 0
        self.to_send = None
        self.to_evaluate = {}
        self.median = 0
        self.evaluations = {}
        self.best_acc = 0
        self.deviations = []
        self.payment_history = []
        self.allocation_history = []

    def __repr__(self):
        return list(string.ascii_uppercase)[self.client_num]

    @property
    def allocation(self):
        return self.test(self.model)[1]

    def send(self):
        """
        Send model to each client in self.receivers and empty own received models.
        """
        for receiver in self.receivers:
            receiver.to_evaluate[self] = self.to_send

    def aggregate(self):
        """
        Aggregate all received models and append models to received models
        """
        new_model = self.architecture
        new_state_dict = {k: 0 for k in new_model.state_dict()}
        for model in self.to_evaluate.values():
            for k, v in model.items():
                new_state_dict[k] += v
        new_state_dict = {
            k: v / len(self.to_evaluate) for k, v in new_state_dict.items()
        }
        return new_state_dict

    def test(self, model):
        self.architecture.load_state_dict(model)
        with torch.no_grad():
            self.architecture.to(self.device)
            self.architecture.load_state_dict(model)
            self.architecture.eval()
            batch_loss, batch_acc = [], []
            for features, labels in self.test_dl:
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                logits = self.architecture(features)
                loss = self.criterion(logits, labels)
                batch_loss.append(loss.cpu())
                pred = torch.argmax(logits, dim=1)
                batch_acc.append(accuracy_score(labels.cpu(), pred.cpu()))
            self.architecture.cpu()
            test_loss = sum(batch_loss) / len(batch_loss)
            test_acc = sum(batch_acc) / len(batch_acc)
            return test_loss, test_acc

    def evaluate(self):
        self.to_send = {}
        for source, model in self.to_evaluate.items():
            _, test_acc = self.test(model)
            source.evaluations[self] = test_acc
            if test_acc > self.best_acc:
                print(f"{self} saves model from {source}, accuracy {test_acc}")
                self.model = model
                self.best_acc = test_acc
                return self.best_acc
        return 0

    def bid(self, deviation=None):
        if deviation:
            self.bid = deviation
        else:
            self.bid = self.value

    def train(self, verbose=False):
        old_model = self.architecture.state_dict()
        self.architecture.load_state_dict(self.model)
        print(f"Training client {self}...")
        self.architecture.to(self.device)
        self.architecture.train()
        batch_loss, batch_acc = [], []
        for features, labels in self.train_dl:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.architecture(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
        if verbose:
            batch_loss.append(loss.cpu())
            pred = torch.argmax(logits, dim=1)
            batch_acc.append(accuracy_score(labels.cpu(), pred.cpu()))
            train_loss = sum(batch_loss) / len(batch_loss)
            train_acc = round(sum(batch_acc) / len(batch_acc), 4)
        self.architecture.cpu()
        model = self.architecture.state_dict()
        if self.test(model)[1] > self.test(old_model)[1]:
            self.model = model
            self.best_acc = self.test(model)[1]
        else:
            self.model = old_model
