import numpy as np
import tensorflow as tf

class ValueDNN():
    def __init__(self, params, obs_dim, logger):
        self.logger = logger
        self.obs_dim = obs_dim
        self.numhiddenlayers = params.critic_hidden_layers
        self.hiddenlayerunits = params.critic_hidden_layer_units

    def BuildNetwork(self, obs_ph):
        self.lr = 1e-2 / np.sqrt(self.hiddenlayerunits[0])  # 1e-3 empirically determined

        out = obs_ph

        for i in range(self.numhiddenlayers):
            self.logger.Debug("hidden layer name", "h" + str(i))
            out = tf.layers.dense(inputs=out, units=self.hiddenlayerunits[i], activation='tanh',
                    kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)),
                                  name=("h" + str(i)))
            self.logger.Debug("out.name", out.name)

        #out = tf.layers.dense(inputs=out, units=1,
        #                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hiddenlayerunits[0])),
        #                           name='output')
        self.value = tf.squeeze(out)
        return self.lr, self.value
