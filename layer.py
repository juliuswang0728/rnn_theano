import numpy as np
import theano
import theano.tensor as T
from theano import function
import operator

class layer_rnn(object):
    #def __init__(self, mini_batch=100, input_dim=28, output_dim=10, hidden_dim=200, n_steps, bptt_truncate=4):
    def __init__(self, n_steps, input_dim, output_dim, hidden_dim):
        # Assign instance variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_steps = np.float32(n_steps)
        np.random.seed(12345)
        #self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (input_dim, hidden_dim))   # 28x200
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, output_dim)) # 200x10
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))    # 200x200
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype('f'))
        self.V = theano.shared(name='V', value=V.astype('f'))
        self.W = theano.shared(name='W', value=W.astype('f'))

        self.define_network()

    def step(self, x_t, s_t_prev, act=T.nnet.softmax):    # x_t: (mini_batch, 28)
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

        cost = T.mean(T.nnet.categorical_crossentropy(mean_states, onehot_y))

        self.get_accuracy = function([x, onehot_y, states], accuracy)
        self.get_cost = function([x, onehot_y, states], cost)
        self.get_prediction = function([x, states], prediction)

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

    def train(self, X, Y, mini_batch, learning_rate, num_epochs, Xv=[], Yv=[]):
        n_train = X.shape[0]
        n_batch = np.int(n_train / mini_batch)
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
                print 'epoch[%d], (tr_acc, val_acc): (%f, %f)' % (epoch_idx,
                self.calc_accuracy(X, Y), self.calc_accuracy(Xv, Yv))

    def calc_accuracy(self, X, Y):
        return self.get_accuracy(X, Y, np.zeros((len(X), self.hidden_dim), dtype='f'))
