from client import Client
import numpy as np
import matplotlib.pyplot as plt


class Server:
    def __init__(
        self, clients: [Client], deviation_pay, transmission_criterion,
    ):
        self.clients = clients
        self.deviation_pay = deviation_pay
        self.transmission_criterion = transmission_criterion

        self.best_acc = 0

    @property
    def client_num(self):
        return len(self.clients)

    @property
    def rounds(self):
        assert len(self.clients[0].payment_history) == len(
            self.clients[0].allocation_history
        )
        return len(self.clients[0].payment_history)

    def circuit_auction(self):
        for client, next_client in zip(
            self.clients, np.random.permutation(self.clients)
        ):
            value = client.bid * (self.best_acc - client.median)
            reserve_price = next_client.bid * (self.best_acc - client.median)
            if value > reserve_price:
                print(
                    f"{client} saves best model, accuracy {self.best_acc}, pays {reserve_price}"
                )
                client.model = self.best_model
                client.pay += reserve_price

    def set_receivers(self):
        for receiver in self.clients:
            for sender in self.clients:
                if sender != receiver and self.transmission_criterion(receiver.bid):
                    sender.receivers.append(receiver)
        for client in self.clients:
            client.receivers.append(
                client
            )  # ensure that there is at least one evaluator; the client itself

    def order_payments(self):
        for client in self.clients:
            client.median = np.median(list(client.evaluations.values()))
            for evaluator in client.evaluations.keys():
                evaluator.deviations.append(
                    client.evaluations[evaluator] - client.median
                )

    def execute_payments(self):
        for client in self.clients:
            client.pay = self.deviation_pay(client.deviations).sum()

    def fed_eval(self):
        for receiver in self.clients:
            acc = receiver.evaluate()
            if acc > self.best_acc:
                self.best_acc = acc
        self.order_payments()
        self.execute_payments()

    def find_best_model(self):
        best_acc = {client: client.best_acc for client in self.clients}
        best_model_owner = max(best_acc, key=best_acc.get)
        self.best_acc = best_model_owner.best_acc
        self.best_model = best_model_owner.model

    def run_demand_auction(self):
        for client in self.clients:
            client.train()
        self.set_receivers()
        for sender in self.clients:
            sender.to_send = sender.model
            sender.send()
        self.fed_eval()
        for client in self.clients:
            client.to_send = client.aggregate()
            client.send()
        self.fed_eval()
        self.find_best_model()
        self.circuit_auction()
        for client in self.clients:
            client.payment_history.append(client.pay)
            client.allocation_history.append(client.allocation)

    def plot(self, what, filename):
        """Testing error for all Demand Clients."""
        assert what in [
            "value",
            "utility",
        ], "Please specify either 'value' or 'utility'"
        y = []
        for client in self.clients:
            plt.plot(
                list(range(self.rounds)),
                np.array(client.allocation_history)
                if what == "value"
                else np.array(client.allocation_history) * client.bid
                - client.payment_history,
                label=f"Client {client} - {client.bid}",
            )
        plt.ylabel(f"{what}")
        plt.xlabel("Federated Rounds")
        plt.grid(True)
        plt.legend()
        plt.savefig(filename, bbox_inches="tight")
        plt.clf()
