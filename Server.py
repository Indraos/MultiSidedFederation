import torch
import numpy as np
import matplotlib.pyplot as plt

class Server:
	def __init__(
		self,
		architecture,
		clients,
		leakage,
		punishment,
		transmission_criterion,
	):
		self.clients = clients
		self.n = len(self.clients)
		self.mbm_acc = []
		self.mbm = []
		self.best_acc = []
		self.local_best_models = []
		self.indiv_acc = []
		self.global_best_model = None
		self.transmission_criterion = transmission_criterion

	#Consecutive Combinations (cyclic)
	def cons_comb(self, clients):
		if self.n%2 == 0:
			return [[clients[i], clients[i+1]] for i in range(len(clients)-1)]
		clients_cyclic = clients + [clients[0]]
		return [[clients_cyclic[i], clients_cyclic[i+1]] for i in range(len(clients_cyclic)-1)]

	def winner(self, client_pair, i):
		client1, client2 = client_pair

		x = client1.bid * (self.best_acc[i] - self.mbm_acc[i])
		y = client2.bid * (self.best_acc[(i+1)%self.n] - self.mbm_acc[(i+1)%self.n])

		if x == y:
			#both get global best model
			client1.model = self.global_best_model
			client2.model = self.global_best_model

		elif x > y:
			#client 1 wins
			client1.model = self.global_best_model
			client2.pay_amt(client1.pay)
			client1.pay_amt(-1 * client1.pay)

		else:
			#client 2 wins
			client2.model = self.global_best_model
			client1.pay_amt(client2.pay)
			client2.pay_amt(-1 * client2.pay)

	def probRecvModel(self,probs):
		n = len(probs)
		probMat = torch.zeros(n,n)
		for i in range(n):
			threshold = np.random.uniform(0.1,1)
			probMat[i] = probs > threshold
		# print(threshold)

		return probMat


	def run_demand_auction(self):
		self.mbm_acc = []
		self.mbm = []
		self.best_acc = []
		self.local_best_models = []
		self.indiv_acc = []

		for client in self.clients:
			client.bidding()

		for client in self.clients:
			probMat = self.probRecvModel(self.transmission_criterion(client.bids))
			for i in range(len(probMat)):
				for j in range(len(probMat)):
					if(probMat[j][i] == 1 and i!=j):
						client.receiver.append(client.clients[j])

		for sender in self.clients:
			sender.send_model()

		for client in self.clients:
			client.evaluate()

			#Getting (median score, median best models(mbm)) and (best score, best model)
			median_score = np.median(client.eval_acc)
			self.mbm_acc.append(median_score)
			best_score = np.max(client.eval_acc)
			self.best_acc.append(best_score)
			
			for acc, model in zip(client.eval_acc, client.evaluated_models):
				if acc == median_score:
					self.mbm.append(model)
					break

			for acc, model in zip(client.eval_acc, client.evaluated_models):
				if acc == best_score:
					self.local_best_models.append(model)
					break

			indiv_score = client.eval_acc[0]
			self.indiv_acc.append(indiv_score)

			amount = np.exp(np.abs(indiv_score - median_score)) + np.exp(np.abs(best_score))
			client.pay = 0
			client.pay_amt(amount)

		self.global_best_model = self.local_best_models[self.best_acc.index(max(self.best_acc))]

		all_combinations = self.cons_comb(self.clients)

		for i,client_pair in enumerate(all_combinations):
			self.winner(client_pair,i)

		"""
		from here: compute median evaluations for all the models that agents produced in the mean-time;
		Run the circuit auction.
		"""

	def print_final_pay(self):
		for client in self.clients:
			print(client.person + ' pays ' + str(client.pay))


	def visualize_values(self,epochs):
		"""Testing error for all Demand Clients.
		"""
		epoch_list = list(range(0,epochs+1))
		for client in self.clients:
			print(client.eval_acc)
			plt.plot(epoch_list, client.eval_acc, label=f'Client {client.person} - {client.bid}')

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
