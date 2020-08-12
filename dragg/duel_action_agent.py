import numpy as np
from sklearn.linear_model import Ridge
import scipy.stats
from dragg.agent import RLAgent

class DuelActionAgent(RLAgent):
    def __init__(self, parameters, rl_log, config):
        RLAgent.__init__(self, parameters, rl_log, config)
        self.action_space = [-0.05, 0.05]
        self.z_theta_mu = [0, 0]
        self.lam_theta = 0.01

    def reward(self):
        return -self.state['curr_error']**2

    def calc_state(self, obs):
        """
        Provides metrics to evaluate and classify the state of the system.
        :return: dictionary
        """
        current_error = (obs.agg_load - obs.agg_setpoint) #/ self.agg_setpoint
        current_rp = obs.reward_price[:2]
        time_of_day = obs.timestep % (24 * obs.dt)
        forecast_error = obs.forecast_load[0][0] - obs.forecast_setpoint
        forecast_trend = obs.forecast_load[0][0] - obs.forecast_load[0][-1]
        std_house = np.std(obs.house_load)

        state = {"curr_error":current_error,
        "time_of_day":time_of_day,
        "fcst_error":forecast_error,
        "forecast_trend": forecast_trend,
        "current_rp": current_rp,
        "whole_forecast_error": np.subtract(obs.forecast_load[0],obs.forecast_setpoint)}
        self.state = state
        return state

    def state_basis(self, state):
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])
        fcst_error_norm = np.linalg.norm(state["whole_forecast_error"])
        forecast_error_norm_basis = np.array([1, fcst_error_norm, fcst_error_norm**2])
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])
        change0_basis = np.array([1, state["current_rp"][0], (state["current_rp"][0])**2])
        change1_basis = np.array([1, state["current_rp"][1], (state["current_rp"][1])**2])
        change_basis = np.outer(change0_basis, change1_basis).flatten()[1:]

        # print(forecast_error_basis, forecast_trend_basis)
        phi = np.outer(forecast_error_basis, forecast_trend_basis).flatten().flatten()[1:]
        # print(phi)
        phi = np.outer(phi, forecast_error_norm_basis).flatten()[1:]
        # print(phi)
        phi = np.outer(phi, time_basis).flatten()[1:]
        # print(phi)
        phi = np.outer(phi, state["current_rp"]).flatten()
        # print(phi)

        return phi

    def state_action_basis(self, state, action):
        action0_basis = np.array([1, action[0], action[0]**2])
        action1_basis = np.array([1, action[1], action[1]**2])
        action_basis = np.outer(action0_basis, action1_basis).flatten()[1:]
        change0_basis = np.array([1, action[0] - state["current_rp"][0], (action[0] - state["current_rp"][0])**2])
        change1_basis = np.array([1, action[1] - state["current_rp"][1], (action[1] - state["current_rp"][1])**2])
        change_basis = np.outer(change0_basis, change1_basis).flatten()[1:]
        action_basis = np.outer(action_basis, change_basis).flatten()[1:]
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])
        fcst_error_norm = np.linalg.norm(state["whole_forecast_error"])
        forecast_error_norm_basis = np.array([1, fcst_error_norm, fcst_error_norm**2])

        v = np.outer(forecast_trend_basis, action_basis).flatten()[1:]
        w = np.outer(forecast_error_basis, action_basis).flatten()[1:] #8
        a = np.outer(forecast_error_norm_basis, action_basis).flatten()[1:]
        b = np.outer(state["whole_forecast_error"], action_basis).flatten()[1:]
        phi = np.concatenate((v, w, a, b))
        phi = np.outer(phi, time_basis).flatten()[1:]

        return phi

    def train(self, env):
        self.next_state = self.calc_state(env)
        if not self.state: # should only be true for timestep 0
            self.state = self.next_state
        if not self.next_action:
            self.next_action = [0,0]
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
        x_k = self.state_basis(state)
        # print(x_k)
        if self.theta_mu is None:
            n = len(x_k)
            self.theta_mu = np.zeros((n, 2))
        # print(self.theta_mu)
        self.mu = x_k @ self.theta_mu
        # print(self.mu)
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])

        action = scipy.stats.norm.rvs(loc=self.mu, scale=self.SIGMA)
        action = np.clip(action, self.actionspace[0], self.actionspace[1])

        return action

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

        self.mu = x_k @ self.theta_mu
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])
        grad_pi_mu = np.zeros(2)
        grad_pi_mu[0] = (self.SIGMA**2) * (self.action[0] - self.mu[0]) * x_k
        grad_pi_mu[1] = (self.SIGMA**2) * (self.action[1] - self.mu[1]) * x_k
        self.z_theta_mu = np.multiply(self.lam_theta,self.z_theta_mu) + (grad_pi_mu)

        self.theta_mu += self.ALPHA_mu * delta * self.z_theta_mu
