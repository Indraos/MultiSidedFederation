class Client:
	client_num = 0
	clients = []

	def __init__(self, model):
		# n-people and their bids
		#['A', 'B', .... 'Z'] -> list of alphabets for visualization purpose
		self.person = list(string.ascii_uppercase)[Client.client_num]
		Client.clients.append(self)
		Client.client_num+=1
		# self.bid = random.random()
		# self.prob = 1 + np.exp(self.bid)
		self.receivers = []
		self.model = model


	def send_model():
		"""
		Send model to each client in self.receivers.
		"""
		for receiver in self.receivers:
			receiver.received_models.append(self.model)


class DemandClient(Client):

	bids = []

	def __init__(self, train_dl, test_dl, optimizer, criterion, initial_model, device):
		super().__init__(initial_model)
		self.device = device
		# self.receivers = []
		self.received_models = []
		self.evaluated_models = []
		self.eval_acc = []
		# self.bid = bid
		self.model = initial_model
		# self.optimizer = Adam(self.model.parameters(), lr = lr)

		

	def bid():
		self.bid = random.random()
		DemandClient.bids.append(bid)
		if len(DemandClient.bids)>len(Client.clients):
			DemandClient.bids = [bid]
		threshold = np.random.uniform(0.1,1)

		return bid, threshold
		# self.prob = 1 + np.exp(self.bid)
		# self.threshold = np.random.uniform(0.1,1)



	def aggregate(self):
		"""
		Aggregate all received models and append models to received models
		"""
		# New Model
		new_model = self.model
		# Average all models
		new_state_dict = {k : 0 for k in new_model.state_dict()}
		for model in self.received_models:
			s_dict = model.state_dict()
			for k, v in s_dict.items():
				new_state_dict[k] +=  v
		new_state_dict = { k : v/len(self.received_models) for k, v in new_state_dict.items()}
		new_model.load_state_dict(new_state_dict)

		return new_model

	def test(self, model):
		with torch.no_grad():
			model.to(self.device)
			model.eval()
			batch_loss, batch_acc = [], []
			for images, labels in self.test_dl:
				if torch.cuda.is_available():
				  images = images.cuda()
				  labels = labels.cuda()
				logits = model(images)
				loss = self.criterion(logits, labels)
				batch_loss.append(loss.cpu())
				pred = torch.argmax(logits, dim=1)
				batch_acc.append(accuracy_score(labels.cpu(), pred.cpu()))
			model.cpu()
			testl = sum(batch_loss)/len(batch_loss)
			testacc = round(sum(batch_acc)/len(batch_acc),4)
			print(f'\nAverage Val Loss: {testl:.4f}, Val Accuracy: {testacc:.4f}\n')
			return testl, testacc

	def evaluate(self):
		"""
		Evaluate all models in received_models and append to evaluated_models
		"""
		for i in range(len(received_models)):
			_ , testacc = self.test(received_models[i])
			eval_acc.append(testacc)
			evaluated_models.append(received_models[i])

		if len(received_models) > 1:
			agg_model = self.aggregate()
			_ , testacc = self.test(agg_model)
			eval_acc.append(testacc)
			evaluated_models.append(agg_model)

	def pay(amount):
		"""
		Change the amount to bid by a value
		"""
		self.pay += amount

	# def send_models():

	# 	return model

	# def bid():



class SupplyClient(Client):
	def __init__(self, train_dl, test_dl, criterion, initial_model):
		self.test_dl = test_dl
		self.train_dl  = train_dl
		self.criterion = criterion
		self.receivers = []

	def train():
		"""
		Train one round using data from the model
		"""
		client_loss, client_acc = [], []
		for model, optimizer, train_dl in zip(client_models, client_optimizers, client_dls):
			# model.load_state_dict(server_model.state_dict())
			closs, cacc = train(model, train_dl, optimizer)
			client_loss.append(closs.item())
			client_acc.append(cacc)
		# return client_loss, client_acc

		return model

	# def send_models():
	# 	"""
	# 	Send models to all clients in receivers.
	# 	"""
	# 	return model


