import numpy as np
import theano
import theano.tensor as T
from theano import function
import cPickle as pickle
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class layer_rnn(object):
	#def __init__(self, mini_batch=100, input_dim=28, output_dim=10, hidden_dim=200, n_steps, bptt_truncate=4):
	def __init__(self, n_steps, input_dim, output_dim, hidden_dim, preload_model=None):
		# Assign instance variables
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.n_steps = np.float32(n_steps)
		self.preload_model = preload_model
		self.test_acc = None
		self.conf_matrix = None

		np.random.seed(12345)
		#self.bptt_truncate = bptt_truncate
		# Randomly initialize the network parameters
		if preload_model is None:
			U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (input_dim, hidden_dim))   # 28x200
			V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, output_dim)) # 200x10
			W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))	# 200x200
		else:
			U, V, W = self.load_model()
		# Theano: Created shared variables
		self.U = theano.shared(name='U', value=U.astype('f'))
		self.V = theano.shared(name='V', value=V.astype('f'))
		self.W = theano.shared(name='W', value=W.astype('f'))

		self.define_network()

	def step(self, x_t, s_t_prev, act=T.nnet.softmax):	# x_t: (mini_batch, 28)
		s_t = T.tanh(T.dot(x_t, self.U) + T.dot(s_t_prev, self.W))  # s_t: (mini_batch, hidden_dim)
		o_t = act(T.dot(s_t, self.V)) # o_t: (mini_batch, output_dim)

		return o_t, s_t

	def define_network(self):
		U, V, W = self.U, self.V, self.W
		x = T.ftensor3('input') # (mini_batch, # of rows, # of columns)
		onehot_y = T.fmatrix('onehot_labels') # (mini_batch, # of classes)
		step_idx = 0

		sum_states = T.zeros((x.shape[0], self.output_dim), dtype='f')
		states = T.zeros((x.shape[0], self.hidden_dim), dtype='f')
		for step_idx in range(0, self.n_steps):
			o_t, states = self.step(x[:, step_idx, :], states)
			sum_states += o_t
		mean_states = sum_states / self.n_steps

		prediction = T.argmax(mean_states, axis=1)
		accuracy = T.mean(T.eq(
			T.argmax(mean_states, axis=1),
			T.argmax(onehot_y, axis=1)
		))
		to_label = T.argmax(onehot_y, axis=1)

		cost = T.mean(T.nnet.categorical_crossentropy(mean_states, onehot_y))

		self.get_accuracy = function([x, onehot_y, states], accuracy)
		self.get_cost = function([x, onehot_y, states], cost)
		self.get_prediction = function([x], prediction)
		self.from_onehot_to_label = function([onehot_y], to_label)

		# Gradients
		dU = T.grad(cost, U)
		dV = T.grad(cost, V)
		dW = T.grad(cost, W)

		# SGD
		learning_rate = T.scalar('learning_rate', dtype='float32')

		self.sgd_step = theano.function([x, onehot_y, learning_rate], [],
					  updates=[(U, U - learning_rate * dU),
							  (V, V - learning_rate * dV),
							  (W, W - learning_rate * dW)])
		self.states = states
		self.sum_states = sum_states

	def train(self, X, Y, mini_batch, learning_rate, num_epochs, evaluation_log='train.log',
				dump_model_name=None, Xv=[], Yv=[], Xt=[], Yt=[]):
		n_train = X.shape[0]
		n_batch = np.int(n_train / mini_batch)
		tr_acc = []
		val_acc = []
		test_acc = None
		for epoch_idx in range(num_epochs):
			perm = np.random.permutation(len(X))
			X = X[perm]
			Y = Y[perm]
			print 'epoch', epoch_idx
			for batch_idx in range(n_batch):
				st = batch_idx * mini_batch
				end = min(st + mini_batch, n_train)
				self.sgd_step(X[st:end], Y[st:end], learning_rate)
			if len(Xv) > 0:
				tr_acc.append(self.calc_accuracy(X, Y).item(0))
				val_acc.append(self.calc_accuracy(Xv, Yv).item(0))
				print 'epoch[%d], (tr_acc, val_acc): (%f, %f)' % (epoch_idx,
				tr_acc[-1], val_acc[-1])

		if len(Xt > 0):
			self.test_acc = self.calc_accuracy(Xt, Yt).item(0)
			print 'test accuracy over %d images: %f' % (len(Xt), self.test_acc)
			pred_test_y = self.get_prediction(Xt)
			self.conf_matrix = confusion_matrix(self.from_onehot_to_label(Yt), pred_test_y)

		self.tr_acc = tr_acc
		self.val_acc = val_acc
		self.dump_model(dump_model_name)
		self.dump_training_log(evaluation_log)

		print 'finished training! Model is dumped in', dump_model_name
		print 'training log is dumped in', evaluation_log

	def calc_accuracy(self, X, Y):
		return self.get_accuracy(X, Y, np.zeros((len(X), self.hidden_dim), dtype='f'))

	def viz_U(self, row_indices):
		U = self.U.get_value()
		n_rows = len(row_indices)
		fig = plt.figure()
		for i, row in enumerate(row_indices, 1):
			a = fig.add_subplot(n_rows, 1, i)
			a.set_title('row %d in U' % row)
			a.set_yticks([])
			U_row = U[:, row].reshape(1, U.shape[0])
			U_row = (U_row - np.min(U_row)) / (np.max(U_row) - np.min(U_row))
			plt.imshow(U_row, cmap='gray', interpolation='nearest')
		plt.tight_layout()
		plt.savefig('viz_U.png')
		plt.show()

	def dump_training_log(self, filename):
		with open(filename, 'wb') as outfile:
			if len(self.tr_acc) > 0:
				pickle.dump(self.tr_acc, outfile, protocol=pickle.HIGHEST_PROTOCOL)
			if len(self.val_acc) > 0:
				pickle.dump(self.val_acc, outfile, protocol=pickle.HIGHEST_PROTOCOL)
			if self.test_acc is not None:
				pickle.dump(self.test_acc, outfile, protocol=pickle.HIGHEST_PROTOCOL)
			if self.conf_matrix is not None:
				pickle.dump(self.conf_matrix, outfile, protocol=pickle.HIGHEST_PROTOCOL)

	def dump_model(self, filename):
		with open(filename, 'wb') as outfile:
			pickle.dump(self.U.get_value(), outfile, protocol=pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.V.get_value(), outfile, protocol=pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.W.get_value(), outfile, protocol=pickle.HIGHEST_PROTOCOL)

	def load_model(self):
		with open(self.preload_model, 'rb') as infile:
			U = pickle.load(infile)
			assert(U.shape[0] == self.input_dim and U.shape[1] == self.hidden_dim)
			V = pickle.load(infile)
			assert(V.shape[0] == self.hidden_dim and V.shape[1] == self.output_dim)
			W = pickle.load(infile)
			assert(W.shape[0] == self.hidden_dim and W.shape[1] == self.hidden_dim)

			return U, V, W
