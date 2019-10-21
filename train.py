import numpy as np
import scipy.signal
from datetime import datetime
import threading
from utils import Logger
from environment import Environment
from utils import  Scaler

class Train(threading.Thread):
    def __init__(self, params, policy, val_func, episodes, env_name, obs_dim, lock, solution):
        threading.Thread.__init__(self)
        self.lock = lock
        self.solution = solution

        date = datetime.utcnow().strftime("%y%m%d")  # create unique directories
        time = datetime.utcnow().strftime("%H%M%S")  # create unique directories
        self.logger = Logger(logname=env_name, date=date, time=time, loglevel=2)
        self.logger.Info(params)

        self.env = Environment(env_name, params, self.logger)
        self.env.reset()
        self.obs_dim = obs_dim

        self.step_size = params.step_size
        self.AllSchedule = []
        self.policy = policy
        self.scaler = Scaler(self.obs_dim)
        self.episodes = episodes
        self.batch_size = params.batch_size
        self.val_func = val_func
        self.lamda = params.lamda
        self.gamma = params.gamma
        self.completeschedules = 0
        self.maxnonzero = 0

    def run_episode(self, env, policy, scaler, animate=False):
        obs = env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        while not done:
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale  # center and scale observations
            observes.append(obs)
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
            self.logger.Debug("next state obs", obs)
            if not isinstance(reward, float):
                reward = np.asscalar(np.asarray(reward))
            rewards.append(reward)
            step += self.step_size  # increment time step feature
        schedule = env.getSchedule()
        self.logger.Debug("schedule", schedule)
        self.AllSchedule.append(schedule)
        return (np.concatenate(observes), np.concatenate(actions),
                np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    def run_policy(self, env, policy, scaler, logger, episodes):
        total_steps = 0
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards, unscaled_obs = self.run_episode(env, policy, scaler)
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'unscaled_obs': unscaled_obs}
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        scaler.update(unscaled)  # update running statistics for scaling observations
        logger.logCSV({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                    'Steps': total_steps})

        return trajectories, self.AllSchedule


    def discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


    def add_disc_sum_rew(self, trajectories, gamma):
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew

    def add_value(self, trajectories, val_func):
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = val_func.predict(observes)
            trajectory['values'] = values

    def add_gae(self, trajectories, gamma, lam):
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages

    def build_train_set(self, trajectories):
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return observes, actions, advantages, disc_sum_rew

    def log_batch_stats(self, observes, actions, advantages, disc_sum_rew, logger, episode):
        """ Log various batch statistics """
        logger.logCSV({'_mean_obs': np.mean(observes),
                    '_min_obs': np.min(observes),
                    '_max_obs': np.max(observes),
                    '_std_obs': np.mean(np.var(observes, axis=0)),
                    '_mean_act': np.mean(actions),
                    '_min_act': np.min(actions),
                    '_max_act': np.max(actions),
                    '_std_act': np.mean(np.var(actions, axis=0)),
                    '_mean_adv': np.mean(advantages),
                    '_min_adv': np.min(advantages),
                    '_max_adv': np.max(advantages),
                    '_std_adv': np.var(advantages),
                    '_mean_discrew': np.mean(disc_sum_rew),
                    '_min_discrew': np.min(disc_sum_rew),
                    '_max_discrew': np.max(disc_sum_rew),
                    '_std_discrew': np.var(disc_sum_rew),
                    '_Episode': episode
                    })

    def GetBestSchedule(self):
        length = len(self.AllSchedule)
        self.logger.Debug("length", length)
        best = self.BestSchedule
        self.logger.Debug("best", best)
        maxbest = max(best)
        minbest = min(best)
        self.logger.Debug("best", best, "maxbest", maxbest)
        foundbest = False
        complete = 0
        for i in range(length):
            nextschedule = self.AllSchedule[i]
            maxslot = max(nextschedule)
            minslot = min (nextschedule)
            nonzero = np.count_nonzero(nextschedule)
            self.logger.Debug("best", best, "maxbest", maxbest, "nextschedule", nextschedule, "maxslot", maxslot, "minslot", minslot)
            if minslot > 0 :
                self.completeschedules += 1
                complete += 1
                if minbest == 0 :
                    best = nextschedule
                    maxbest = maxslot
                    foundbest = True
                if maxslot < maxbest:
                    best = nextschedule
                    maxbest = maxslot
                    foundbest = True
            else :
                if nonzero > self.maxnonzero:
                    self.maxnonzero = nonzero
                    best = nextschedule
                    maxbest = maxslot
                elif nonzero == self.maxnonzero:
                    if maxslot < maxbest:
                        best = nextschedule
                        maxbest = maxslot
        self.BestSchedule = best
        self.AllSchedule = []
        return best, foundbest, complete

    def run(self):
        startTime = datetime.now()
        self.logger.Critical("start time", startTime)
        self.run_policy(self.env, self.policy, self.scaler, self.logger, episodes=5)
        self.BestSchedule = self.env.getSchedule()
        episode = 0
        while episode < self.episodes:
            trajectories, allSchedule = self.run_policy(self.env, self.policy, self.scaler, self.logger, episodes=self.batch_size)
            self.AllSchedule = allSchedule
            episode += len(trajectories)
            self.add_value(trajectories, self.val_func)  # add estimated values to episodes
            self.add_disc_sum_rew(trajectories, self.lamda)  # calculated discounted sum of Rs
            self.add_gae(trajectories, self.lamda, self.gamma)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = self.build_train_set(trajectories)
            # add various stats to training log:
            self.log_batch_stats(observes, actions, advantages, disc_sum_rew, self.logger, episode)
            schedule, foundbest, completeschedules = self.GetBestSchedule()
            maxslot = max(schedule)
            self.lock.acquire()
            #globalmaxnonzero, globalmaxslot = self.solution.get()
            #if self.maxnonzero > globalmaxnonzero or maxslot < globalmaxslot:
                #updated = True
            self.solution.update(self.maxnonzero, maxslot)
            self.policy.update(observes, actions, advantages, self.logger)  # update policy
            self.val_func.fit(observes, disc_sum_rew, self.logger)  # update value function
            #else:
                #updated = False
            self.lock.release()
            #if updated:
            self.logger.write(display=True)  # write logger results to file and stdout
            self.logger.Critical("slots", self.maxnonzero, "max", maxslot, "complete", completeschedules, "total",
                                 self.completeschedules)#, "update", updated)
            #else:
                #self.logger.Critical("Episode", episode, "slots", self.maxnonzero, "max", maxslot, "complete",
                                     #completeschedules, "total", self.completeschedules, "update", updated)
        maxslot = max(self.BestSchedule)
        nonzero = np.count_nonzero(self.BestSchedule)
        self.logger.Critical('Best', self.BestSchedule, "nonzero", nonzero, "max", maxslot, "complete", self.completeschedules)
        self.logger.Critical("Time taken:", datetime.now() - startTime)
