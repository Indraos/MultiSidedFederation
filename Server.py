class Server:
    def __init__(
        self,
        architecture,
        supply_clients,
        demand_clients,
        leakage,
        punishment,
        transmission_criterion,
    ):
        self.estimate_single = np.zeros((len(self.supply_clients, self.demand_clients)))

    def run_demand_auction():
        for client in self.demand_clients:
            client.bid()
        for sender in set(self.supply_clients).union(self.demand_client):
            for receiver in self.demand_clients:
                if self.transmission_criterion(receiver.bid()):
                    sender.receivers.append(receiver)
        for sender in self.supply_clients:
            sender.send_models()
        for receiver in self.demand_clients:
            receiver.evaluate()
            receiver.aggregate()
            receiver.send_models()
        for receiver in self.demand_clients:
            receiver.evaluate()
        np.median(self.estimate_single, axis=1)

    def visualize_values():
        """Testing error for all Demand Clients.
        """
        pass

    def visualize_utilities():
        """Objective functions for all demand clients.
        """
        pass
