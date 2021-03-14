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
        """Constructor

        Args:
            clients ([Client]): Clients participating
            deviation_pay (Function): function mapping deviation to monetary deviation_pay
            transmission_criterion (Function): Function determining local transmission
        """
        self.clients = clients
        self.deviation_pay = deviation_pay
        self.transmission_criterion = transmission_criterion

        self.mbm_acc = []
        self.mbm = []
        self.local_best_models = []
        self.best_model_acc = []
        self.best_model = None

    @property
    def client_num(self):
        return len(self.clients)

    def client_pairs(self, clients):
        return zip(
            range(self.client_num), np.random.permutation(list(range(self.client_num)))
        )

    def winner(self, client_pair, i):
        client1, client2 = client_pair

        value = client.bid * (self.best_model_acc - client.median_acc)
        reserve_price = next_client.bid * (self.best_model_acc - next_client.median_acc)

        if value > reserve_price:
            client1.model.load_state_dict(self.best_model.state_dict())
            client2.pay_amt(reserve_price)

    def probRecvModel(self, probs):
        n = len(probs)
        probMat = torch.zeros(n, n)
        for i in range(n):
            threshold = np.random.uniform(0.1, 1)
            probMat[i] = probs > threshold
        return probMat

    def run_demand_auction(self):
        self.mbm_acc = []
        self.mbm = []
        self.best_model_acc = []
        self.local_best_models = []
        self.indiv_acc = []

        for client in self.clients:
            client.bid()

        # print(f"Client bids = {client.bids}")
        # print(f"Probs = {1-np.exp(-1 * [1,1,2])}")

        # print(f"Probs = {self.transmission_criterion(client.bids)}")

        probMat = self.probRecvModel(self.transmission_criterion(client.bids))
        probMat = torch.transpose(probMat, 0, 1)
        # print(probMat)

        for i, client in enumerate(self.clients):
            # print(self.transmission_criterion(client.bids))

            prob = probMat[i]
            for j in range(len(probMat)):
                if prob[j] == 1 and i != j:
                    # print(f"({i},{j})")
                    client.receivers.append(client.clients[j])
                    # print(client.person + ' receiving ' + client.clients[j].person)

        # print(probMat)

        for sender in self.clients:
            sender.send_model()

        for client in self.clients:
            print(f"\nClient {client.person}: ")
            print("******")
            client.evaluate()

            # Getting (median score, median best models(mbm)) and (best score, best model)
            median_score = np.median(client.eval_acc)
            self.mbm_acc.append(median_score)
            best_score = np.max(client.eval_acc)
            self.best_model_acc.append(best_score)

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

            amount += self.deviation_pay(self.indiv_score - self.median_score)
            amount += self.deviation_pay(self.best_score - self.median_score)
            client.pay_amt(amount)
        self.best_model = self.local_best_models[
            self.best_model_acc.index(max(self.best_model_acc))
        ]

        for i, client_pair in enumerate(self.client_pairs()):
            self.winner(client_pair, i)

    def visualize_values(self):
        """Testing error for all Demand Clients."""
        rounds = list(range(len(self.val_acc)))
        for client in self.clients:
            plt.plot(
                epoch_list,
                client.val_acc,
                label=f"Client {client.person} - {client.bid}",
            )

        plt.ylabel("Accuracy Value")
        plt.xlabel("Federated Rounds")
        plt.title("Client Models Accuracy", fontsize=18)
        plt.grid(True)
        plt.legend(
            title="Model - Bid Value", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.show()

    def visualize_utilities():
        """Objective for all Demand Clients."""
        rounds = list(range(len(self.val_acc)))
        for client in self.clients:
            plt.plot(
                rounds,
                client.val_acc * client.bid - client.pay,
                label=f"Client {client.person} - {client.bid}",
            )

        plt.ylabel("Objective Value")
        plt.xlabel("Federated Rounds")
        plt.title("Client Objective", fontsize=18)
        plt.grid(True)
        plt.legend(
            title="Model - Bid Value", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.show()