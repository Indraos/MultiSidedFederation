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
		self.estimate_aggregated = np.zeros((len(self.demand_clients, self.demand_clients)))

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
		"""
		from here: compute median evaluations for all the models that agents produced in the mean-time;
		Run the circuit auction.
		"""
		np.median(self.estimate_single, axis=1)

	def visualize_values(self,epochs):
		"""Testing error for all Demand Clients.
		"""
		epoch_list = list(range(0,epochs+1))
		for client in self.demand_clients:
		  plt.plot(epoch_list, client.eval_acc, label=f'Client {client.person} - {client.bid:4f}')

		plt.ylabel('Accuracy Value')
		plt.xlabel('Federated Rounds')
		plt.title('Client Models Accuracy', fontsize = 18)
		plt.grid(True)
		l1 = plt.legend(title = 'Model - Bid Value',bbox_to_anchor=(1.05, 1),loc="upper left")
		plt.show()


	def visualize_utilities():
		"""Objective functions for all demand clients.
		"""
		pass
