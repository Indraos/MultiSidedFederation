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

        self.best_accuracy = 0

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
            value = client.bid * (self.best_accuracy - client.median)
            reserve_price = next_client.bid * (self.best_accuracy - next_client.median)
            if value > reserve_price:
                client1.model.load_state_dict(self.best_model.state_dict())
                client2.pay += reserve_price

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
            receiver.evaluate()
        self.order_payments()
        self.execute_payments()

    def find_best_model(self):
        best_acc = {client: client.best_acc for client in self.clients}
        best_model_owner = max(best_acc, key=best_acc.get)
        self.best_acc = best_model_owner.best_acc
        self.best_model = best_model_owner.architecture

    def run_demand_auction(self):
        self.set_receivers()
        for sender in self.clients:
            sender.to_send = sender.architecture
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
            client.pay = 0

    def plot(self, what, filename):
        """Testing error for all Demand Clients."""
        rounds = self.rounds
        assert what in ["values", "utilities"], "only support utilities and values."
        for client in self.clients:
            if what == "values":
                y = client.median
            if what == "utilities":
                y = client.median * client.bid - client.pay
        plt.plot(
            rounds, y, label=f"Client {client} - {client.bid}",
        )
        plt.ylabel(f"{what} Value")
        plt.xlabel("Federated Rounds")
        plt.title(f"Client Models {what}", fontsize=18)
        plt.grid(True)
        plt.legend(
            title="Model - Bid Value", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.savefig(filename)
