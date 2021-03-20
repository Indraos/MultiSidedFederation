from client import Client
import numpy as np
import matplotlib.pyplot as plt


class Server:
    def __init__(
        self,
        clients: [Client],
        deviation_pay,
        transmission_criterion,
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
        assert len(self.pay_history[0]) == len(self.allocation_history[0])
        return len(self.pay_history[0])

    def circuit_auction(self):
        for client, next_client in zip(
            range(self.client_num), np.random.permutation(list(range(self.client_num)))
        ):
            value = client.bid * (self.best_accuracy - client.median_accuracy)
            reserve_price = next_client.bid * (
                self.best_accuracy - next_client.median_accuracy
            )
            if value > reserve_price:
                client1.model.load_state_dict(self.best_model.state_dict())
                client2.pay += reserve_price

    def set_receivers(self):
        for sender in self.clients:
            for receiver in self.clients:
                if self.transmission_criterion(receiver.bid):
                    sender.receivers.append(receiver)

    def order_payment(self):
        for client in self.clients:
            client.median = np.median(evaluations.items())
            for evaluator in client.evaluations.keys():
                evaluator.deviations.append(
                    client.evaluations[evaluator] - client.median
                )

    def payment(self):
        for client in self.clients:
            client.pay = self.deviation_pay(evaluators.deviations).sum()

    def fed_eval(self):
        for receiver in self.clients:
            receiver.evaluate()
        self.order_payments()
        self.pay()

    def best_model(self):
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
            client.aggregate()
            client.to_send(self.aggregated_model)
            client.send()
        self.fed_eval()
        self.best_model()
        self.circuit_auction(client_pair, i)
        for client in self.clients:
            client.payment_history.append(client.pay)
            client.allocation_history.append(client.allocation)
            client.pay = 0

    def plot(self, what, filename):
        """Testing error for all Demand Clients."""
        rounds = list(range(len(self.val_acc)))
        assert what in ["values", "utilities"], "only support utilities and values."
        for client in self.clients:
            if what == "values":
                y = client.median_values
            if what == "utilities":
                y = client.median_values * client.bid - client.pay
        plt.plot(
            rounds,
            y,
            label=f"Client {client.person} - {client.bid}",
        )
        plt.ylabel(f"{what} Value")
        plt.xlabel("Federated Rounds")
        plt.title(f"Client Models {what}", fontsize=18)
        plt.grid(True)
        plt.legend(
            title="Model - Bid Value", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.save(filename)