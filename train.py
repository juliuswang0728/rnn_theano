from data_handler import load_data, onehot
from layer import layer_rnn
import numpy as np
import theano
from util import load_training_log, plot_confusion_matrix

theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'

params = []

# Read MNIST training set, validation set, and test set
(X, Y), (Xv, Yv), (Xt, Yt) = load_data('mnist.pkl.gz')
Y = onehot(Y)
Yv = onehot(Yv)
Yt = onehot(Yt)

input_dim = X.shape[2]
output_dim = Y.shape[1]
hidden_dim = 200
mini_batch = 100
num_epochs = 100
lr = np.float32(0.01)
n_steps = X.shape[1]

# define theano network
rnn = layer_rnn(n_steps=n_steps,
                input_dim=X.shape[2], output_dim=Y.shape[1],
                hidden_dim=hidden_dim)

rnn.train(X, Y, mini_batch=mini_batch, learning_rate=lr, num_epochs=num_epochs,
            dump_model_name='rnn.model', Xv=Xv, Yv=Yv, Xt=Xt, Yt=Yt)

#rnn.viz_U([0, 1, 2, 6, 9]) # visualize rows in U matrix by specifying the row indices

#tr_acc, val_acc, test_acc, conf_matrix = load_training_log()
#plot_confusion_matrix(conf_matrix)
