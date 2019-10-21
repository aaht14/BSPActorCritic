import tensorflow as tf
import numpy as np

class PolicyDNN():
    def __init__(self, params, obs_dim, act_dim, logger):
        self.logger = logger
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy_logvar = params.actor_policy_logvar
        self.numhiddenlayers = params.actor_hidden_layers
        self.hiddenlayerunits = params.actor_hidden_layer_units
        self.counter = 1

    def BuildNetwork(self, obs_ph):
        self.lr = 9e-4 / np.sqrt(self.hiddenlayerunits[0])  # 9e-4 empirically determined

        out = obs_ph

        for i in range(self.numhiddenlayers):
            self.logger.Debug("hidden layer name", "h" + str(i))
            if i < self.numhiddenlayers - 1:
                print("out", out)
                self.logger.Debug("not last hidden layer")
                out = tf.keras.layers.LSTM(units=self.hiddenlayerunits[i], return_sequences=True, activation = 'tanh',
                   kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name=("h" + str(i)))(out)
            else :
                self.logger.Debug("last hidden layer")
                self.means = tf.keras.layers.LSTM(units=self.hiddenlayerunits[i], return_sequences=False, activation = 'tanh',
                   kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name=("h" + str(i)))(out)
                #self.means = tf.layers.dense(inputs=out, units=self.act_dim,
                #               kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hiddenlayerunits[0])), name="means")

        #self.means = tf.layers.dense(inputs=out, units=self.act_dim,
                           #kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hiddenlayerunits[0])), name="means")

        print("self.means", self.means)

        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.

        logvar_speed = (10 * self.hiddenlayerunits[0]) // 48

        log_vars = tf.get_variable('logvars' + str(self.counter), (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))

        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        return self.lr, self.log_vars, self.means
