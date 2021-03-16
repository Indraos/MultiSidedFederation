import torch
import numpy as np
import matplotlib.pyplot as plt


class Server:
    def __init__(
        self,
        clients,
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
        return len(self.clients[0].median_accuracy)

    def client_pairs(self, clients):
        return zip(
            range(self.client_num), np.random.permutation(list(range(self.client_num)))
        )

    def winner(self, client, next_client):
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

    def run_demand_auction(self):
        self.mbm_acc = []
        self.mbm = []
        self.best_model_acc = []
        self.local_best_models = []
        self.indiv_acc = []

        self.set_receivers()
        for sender in self.clients.values():
            self.to_send.append(self.model)
            client.send()
        for key, receiver in self.clients().items():
            self.evaluate(key)
        for client in self.clients():
            client.median_score = np.median(np.array(client.scores_single.values()))
            for identity, evaluator in self.clients().items():
                evaluator.pay += self.deviation_pay(
                    client.evaluator_scores[identity] - client.median_score
                )
        for client in self.clients():
            client.aggregate()
            client.to_send(self.aggregated_model)
            client.send()
        for key, receiver in self.clients().items():
            self.evaluate(key)
        for client in self.clients():
            client.median_score = np.median(np.array(client.scores_single.values()))
            for identity, evaluator in self.clients().items():
                evaluator.pay += self.deviation_pay(
                    client.evaluator_scores[identity] - client.median_score
                )

            for acc, model in zip(client.eval_acc, client.evaluated_models):
                if acc == median_score:
                    self.mbm.append(model)
                    break

            for acc, model in zip(client.eval_acc, client.evaluated_models):
                if acc == best_score:
                    self.local_best_models.append(model)
                    break

            indiv_score = client.eval_acc[0]
            client.indiv_acc = indiv_score
            client.pay_amt(amount)
        self.best_model = self.local_best_models[
            self.best_model_acc.index(max(self.best_model_acc))
        ]

        for i, client_pair in enumerate(self.client_pairs()):
            self.winner(client_pair, i)

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