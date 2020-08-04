import os
import sys
import threading
from queue import Queue
from copy import deepcopy

import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
import toml
import random
import names
import string
import cvxpy as cp
import dccp
import itertools as it
import redis
from sklearn.linear_model import Ridge
import scipy.stats
from abc import ABC, abstractmethod

# Local
from dragg.mpc_calc import MPCCalc
from dragg.redis_client import RedisClient
from dragg.logger import Logger

class RLAgent(ABC):
    def __init__(self, parameters, rl_log, config):
        # self.name = name
        # self.env = environment
        self.data_dir = 'data'
        # self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
        # self.config = self._import_config()
        self.config = config
        self.actionspace = self.config['rl']['utility']['action_space']
        self.theta_mu = None
        self.theta_q = None
        self.prev_state = None
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.memory = []
        self.cumulative_reward = 0
        self.average_reward = 0
        self.mu = 0
        self.rla_log = rl_log
        self.i = 0
        self.z_theta_mu = 0
        self.z_theta_sigma = 0
        self.lam_theta = 0.01

        self.rl_data = {} #self.set_rl_data()
        self.set_rl_data()
        self._set_parameters(parameters)

    @abstractmethod
    def calc_state(self, env):
        pass

    def _import_config(self):
        with open(self.config_file) as f:
            data = toml.load(f)

        self.actionspace = data['rl']['utility']['action_space'] # this is janky
        return data

    def _set_parameters(self, params):
        self.ALPHA_q = params['alpha']
        self.ALPHA_mu = params['alpha']
        self.ALPHA_w = params['alpha'] * (2)
        self.ALPHA_r = params['alpha'] * (2 ** 2)
        self.BETA = params['beta']
        self.BATCH_SIZE = params['batch_size']
        self.TWIN_Q = params['twin_q']
        self.SIGMA = params['epsilon']

    def state_basis(self, state):
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])

        phi = np.outer(forecast_error_basis, forecast_trend_basis).flatten()[1:]
        phi = np.outer(phi, time_basis).flatten()[1:]

        return phi

    def state_action_basis(self, state, action):
        # action_basis = np.array([1, (action), (action)**2, ((action - 0.02))**2, ((action + 0.02))**2])
        action_basis = np.array([1, action, action**2])
        # change_rp = self.state['reward_price'][0] - self.state['reward_price'][-1]
        delta_action_basis = np.array([1, state['delta_action'], state['delta_action']**2])
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])
        # curr_error_basis = np.array([1, state["curr_error"], state["curr_error"]**2])
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])

        # v = np.outer(avg_forecast_error_basis, action_basis).flatten()[1:] #14 (indexed to 13)
        # w = np.outer(curr_error_basis, delta_action_basis).flatten()[1:]
        v = np.outer(forecast_trend_basis, action_basis).flatten()[1:]
        w = np.outer(forecast_error_basis, action_basis).flatten()[1:] #8
        z = np.outer(forecast_error_basis, delta_action_basis).flatten()[1:] #14
        phi = np.concatenate((v, w, z))
        phi = np.outer(phi, time_basis).flatten()[1:]

        # phi = np.clip(phi, -100, 150)
        return phi

    @abstractmethod
    def reward(self):
        """ Reward function encourages the RL agent to move towards a
        state with curr_error = 0. Negative reward values ensure that the agent
        tries to terminate the "epsiode" as soon as possible.
        _reward() should only be called to calculate the reward at the current
        timestep, when reward must be used in stochastic gradient descent it may
        be sampled through an experience tuple.
        :return: float
        """
        pass
        # return -self.state["curr_error"]**2 #+ 10**3 * sum(np.square(self.reward_price))
        # return -self.next_state["curr_error"]**2

    def memorize(self):
        if self.state and self.action:
            experience = {"state": self.state, "action": self.action, "reward": self.r, "next_state": self.next_state}

    def train(self, env):
        self.next_state = self.calc_state(env)
        if not self.state: # should only be true for timestep 0
            self.state = self.next_state
        if not self.next_action:
            self.next_action = 0
        self.action = self.next_action

        self.r = self.reward()

        self.xu_k = self.state_action_basis(self.state, self.action)
        self.next_action = self.get_policy_action(self.next_state)
        self.xu_k1 = self.state_action_basis(self.next_state, self.next_action)
        self.memorize()
        self.update_qfunction()
        self.update_policy()
        self.record_rl_data()

        self.state = self.next_state
        return self.next_action

    def get_policy_action(self, state):
        """
        Selects action of the RL agent according to a Gaussian probability density
        function.
        Gaussian mean is parameterized linearly. Gaussian standard deviation is fixed.
        :return: float
        """
        pi = []
        x_k = self.state_basis(state)
        if self.theta_mu is None:
            n = len(x_k)
            self.theta_mu = np.zeros(n)
        self.mu = self.theta_mu @ x_k
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])

        action = scipy.stats.norm.rvs(loc=self.mu, scale=self.SIGMA)
        action = np.clip(action, self.actionspace[0], self.actionspace[1])

        return action

    def update_qfunction(self):
        if self.TWIN_Q:
            self.i = (self.i + 1) % 2

        if self.theta_q is None: # generate critic network if none exist
            n = len(self.state_action_basis(self.state, self.action))
            if self.TWIN_Q:
                m = 2 # generate 2 q networks
            else:
                m = 1
            self.theta_q = np.random.normal(-1, 0.3, (n, m))
        self.q_predicted = self.theta_q[:,self.i] @  self.xu_k # recorded for analysis
        self.q_observed = self.r + self.BETA * self.theta_q[:,self.i] @ self.xu_k1 # recorded for analysis
        # temp_theta = deepcopy(self.theta)

        if len(self.memory) > self.BATCH_SIZE:
            batch = random.sample(self.memory, self.BATCH_SIZE)
            batch_y = []
            batch_phi = []
            for exp in batch:
                x = exp["state"]
                x1 = exp["next_state"]
                u = exp["action"]
                u1 = self.get_policy_action(x1)
                xu_k = self.state_action_basis(x,u)
                xu_k1 = self.state_action_basis(x1,u1)
                q_k1 = min(self.theta_q[:,i] @ xu_k1 for i in range(len(q)))
                y = exp["reward"] + self.BETA * q_k1
                # y = exp["reward"] + self.BETA * q1_a
                batch_y.append(y)
                batch_phi.append(xu_k)
            batch_y = np.array(batch_y)
            batch_phi = np.array(batch_phi)

            if np.isnan(batch_y).any():
                if np.isnan(self.theta_q).any():
                    self.rla_log.logger.error("Q network has diverged and has caused Q value predictions to be infinite.")
            if np.isnan(batch_phi).any():
                self.rla_log.logger.error("Numerical innaccuracies in the state basis calculation.")

            clf = Ridge(alpha = 0.01)
            clf.fit(batch_phi, batch_y)
            temp_theta = clf.coef_
            self.theta_q[:,self.i] = self.ALPHA_q * temp_theta + (1-self.ALPHA_q) * theta

    def update_policy(self):
        """
        Updates the mean of the Gaussian action selection policy.
        :return:
        """
        x_k = self.state_basis(self.state)
        x_k1 = self.state_basis(self.next_state)
        delta = self.q_predicted - self.q_observed #- self.average_reward
        delta = np.clip(delta, -1, 1)
        self.average_reward += self.ALPHA_r * delta
        self.cumulative_reward += self.r
        # self.average_reward = self.cummulative_reward / (self.timestep + 1)
        # self.z_w = self.lam_w * self.z_w + (x_k1 - x_k)
        self.mu = self.theta_mu @ x_k
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])
        grad_pi_mu = (self.SIGMA**2) * (self.action - self.mu) * x_k
        self.z_theta_mu = self.lam_theta * self.z_theta_mu + (grad_pi_mu)
        # self.w += self.ALPHA_w * delta * self.z_w # update reward function
        self.theta_mu += self.ALPHA_mu * delta * self.z_theta_mu

    def set_rl_data(self):
        # temp = {}
        self.rl_data["theta"] = []
        self.rl_data["phi"] = []
        self.rl_data["q_obs"] = []
        self.rl_data["q_pred"] = []
        self.rl_data["action"] = []
        self.rl_data["q_tables"] = []
        self.rl_data["average_reward"] = []
        self.rl_data["cumulative_reward"] = []
        self.rl_data["reward"] = []
        self.rl_data["mu"] = []

    def record_rl_data(self):
        self.rl_data["theta"].append(self.theta_q[:,self.i].flatten().tolist())
        self.rl_data["q_obs"].append(self.q_observed)
        self.rl_data["q_pred"].append(self.q_predicted)
        self.rl_data["action"].append(self.action)
        # self.rl_data["is_greedy"].append(self.is_greedy)
        self.rl_data["average_reward"].append(self.average_reward)
        self.rl_data["cumulative_reward"].append(self.cumulative_reward)
        self.rl_data["reward"].append(self.r)
        self.rl_data["mu"].append(self.mu)

    # @staticmethod
    def write_rl_data(self, output_dir):
        file = os.path.join(output_dir, f"{self.name}_agent-results.json")
        with open(file, "w+") as f:
            json.dump(self.rl_data, f, indent=4)
