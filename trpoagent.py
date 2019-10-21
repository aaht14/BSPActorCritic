from datetime import datetime
import tensorflow as tf
from utils import GracefulKiller
from critic import Critic
from actor import Actor
from train import Train
from utils import Logger
import threading
from environment import Solution
import numpy as np

class TRPOAgent():
    def __init__(self, params, env_name, episodes):

        self.killer = GracefulKiller()

        tf.reset_default_graph()
        seed = params.seed * 1958
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.episodes = episodes
        self.batch_size = params.batch_size
        self.lamda = params.lamda
        self.gamma = params.gamma
        self.env_name = env_name
        self.params = params
        self.threads = params.threads
        self.obs_dim = params.obs_dim
        self.act_dim = params.act_dim
        self.obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
        date = datetime.utcnow().strftime("%y%m%d")  # create unique directories
        time = datetime.utcnow().strftime("%H%M%S")  # create unique directories
        self.lock = threading.Lock()
        self.solution = Solution()

        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # 1 Debug 2 Info 3 Warning 4 Error 5 Critical
        self.logger = Logger(logname=env_name, date=date, time=time, loglevel=2)
        self.logger.Info(params)

        self.policy = Actor(params, self.obs_dim, self.act_dim, self.config, self.logger)
        self.val_func = Critic(params, self.obs_dim, self.config, self.logger)

    def run(self):

        self.train = [Train(self.params, self.policy, self.val_func, self.episodes, self.env_name, self.obs_dim,
                            self.lock, self.solution)
                      for i in range(self.threads)]

        for agent in self.train:
            agent.start()

        for agent in self.train:
            agent.join()

        self.close_sess()

    def close_sess(self):
        self.logger.close()
        self.policy.close_sess()
        self.val_func.close_sess()
