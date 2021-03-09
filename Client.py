class Client:
    def __init__():
        self.receivers = []
        pass

    def send_model():
        """
        Send model to each client in self.receivers.
        """


class DemandClient(Client):
    def __init__(self, train_dl, optimizer, test_dl, criterion, initial_model, bid):
        self.__super__()
        self.receivers = []
        self.received_models = []
        self.evaluated_models = []
        self.bid = bid

    def aggregate():
        """
        Aggregate all received models and append models to received models
        """
        return model

    def evaluate():
        """
        Evaluate all models in received_models and append to evaluated_models
        """
        pass

    def pay(amount):
        """
        Change the amount to bid by a value
        """
        self.pay += amount

    def send_models():
        return model


class SupplyClient(Client):
    def __init__(self, test_dl, criterion, initial_model):
        self.test_dl = test_dl
        self.criterion = criterion
        self.receivers = []

    def train():
        """
        Train one round using data from the model
        """
        return model

    def send_models():
        """
        Send models to all clients in receivers.
        """
        return model


