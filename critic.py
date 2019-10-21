import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from valuednn import ValueDNN

class Critic(object):
    def __init__(self, params, obs_dim, config, logger):
        self.device = params.device
        self.logger = logger
        self.config = config
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        #self.hid1_mult = hid1_mult
        self.epochs = params.critic_epochs
        self.lr = None  # learning rate set in _build_graph()
        self.value_dnn = ValueDNN(params, self.obs_dim, logger)
        self._build_graph()
        self._init_session()

    def _init_session(self):
        self.sess = tf.Session(config=self.config, graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        with tf.device(self.device):
            self.g = tf.Graph()
            with self.g.as_default():
                self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
                self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')

                self.lr, self.value = self.value_dnn.BuildNetwork(self.obs_ph)

                self.loss = tf.reduce_mean(tf.square(self.value - self.val_ph))  # squared loss
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.minimize(self.loss)
                self.init = tf.global_variables_initializer()

    def fit(self, x, y, logger):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        #print("x.shape", x.shape, "y.shape", y.shape, "self.replay_buffer_x.shape", self.replay_buffer_x.shape,
              #"self.replay_buffer_y.shape", self.replay_buffer_y.shape, "x_train.shape", x_train.shape,
              #"y_train.shape", y_train.shape, "num_batches", num_batches, "batch_size", batch_size)
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.logCSV({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.value, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
