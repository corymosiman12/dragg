import numpy as np
from sklearn.linear_model import Ridge
import scipy.stats
from agent import RLAgent

class DualActionAgent(RLAgent):
    def __init__(self, parameters, rl_log, num_actions=2):
        RLAgent.__init__(self, parameters, rl_log)
        self.action_space = [-0.05, 0.05]
        self.z_theta_mu = [0, 0]
        self.lam_theta = 0.01
        self.num_actions = num_actions

    def reward(self):
        return -self.state['curr_error']**2

    def state_basis(self, state):
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])

        phi = np.outer(forecast_error_basis, forecast_trend_basis).flatten()[1:]
        phi = np.outer(phi, time_basis).flatten()[1:]

        return phi

    def state_action_basis(self, state, action):
        action_basis = [1]
        for i in range(self.num_actions):
            for j in range(i, self.num_actions):
                a = [1, action[i], action[i]**2]
                b = [1, action[j], action[j]**2]
                action_basis += np.outer(a,b)[1:].flatten().tolist()
        action_basis = np.array(action_basis)
        delta_action_basis = np.array([1, state['delta_action'], state['delta_action']**2])
        time_basis = np.array([1, np.sin(2 * np.pi * state["time_of_day"]), np.cos(2 * np.pi * state["time_of_day"])])
        forecast_error_basis = np.array([1, state["fcst_error"], state["fcst_error"]**2])
        forecast_trend_basis = np.array([1, state["forecast_trend"], state["forecast_trend"]**2])

        v = np.outer(forecast_trend_basis, action_basis).flatten()[1:]
        w = np.outer(forecast_error_basis, action_basis).flatten()[1:]
        z = np.outer(forecast_error_basis, delta_action_basis).flatten()[1:]
        phi = np.concatenate((v, w, z))
        phi = np.outer(phi, time_basis).flatten()[1:]

        return phi

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
        x_k = self.state_basis(state)
        if self.theta_mu is None:
            n = len(x_k)
            self.theta_mu = np.zeros((n, self.num_actions))
        self.mu = x_k @ self.theta_mu
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])

        action = scipy.stats.norm.rvs(loc=self.mu, scale=self.SIGMA)
        action = np.clip(action, self.actionspace[0], self.actionspace[1])

        return action.tolist()

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
                q_k1 = min(xu_k1 @ self.theta_q[:,i] for i in range(len(q)))
                y = exp["reward"] + self.BETA * q_k1
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
        self.mu = self.theta_mu @ x_k
        self.mu = np.clip(self.mu, self.actionspace[0], self.actionspace[1])
        grad_pi_mu = (self.SIGMA**2) * (self.action - self.mu) * x_k
        self.z_theta_mu = self.lam_theta * self.z_theta_mu + (grad_pi_mu)
        self.theta_mu += self.ALPHA_mu * delta * self.z_theta_mu
