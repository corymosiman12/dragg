import numpy as np

class RandomAgent():
    def __init__(self, env):
        self.env = env

    def predict(state):
        """
        :input: state observation vector
        :return: action in the form of a list of same length as the environment action space
        """
        action = self.env.action_space.sample()
        return action
