import numpy as np
from dragg.agent import RLAgent

class HorizonAgent(RLAgent):
    def __init__(self, parameters, rl_log):
        RLAgent.__init__(self, parameters, rl_log)
        self.name = "horizon"

    def reward(self):
        return -1*((10*self.state['curr_error'])**2)

    def calc_state(self, obs):
        """
        Provides metrics to evaluate and classify the state of the system.
        :return: dictionary
        """
        current_error = (obs.agg_load - obs.agg_setpoint)
        change_rp = obs.reward_price[0] - obs.reward_price[-1]
        time_of_day = obs.timestep % (24 * obs.dt)
        forecast_error = obs.forecast_load[0] - obs.forecast_setpoint
        forecast_trend = obs.forecast_load[0] - obs.forecast_load[-1]

        state = {"curr_error":current_error,
                "time_of_day":time_of_day,
                "fcst_error":forecast_error,
                "forecast_trend": forecast_trend,
                "delta_action": change_rp}
        self.state = state
        return state

class NextTSAgent(RLAgent):
    def __init__(self, parameters, rl_log, timestep):
        RLAgent.__init__(self, parameters, rl_log)
        self.name = "next"
        self.ts = timestep
        self.total_p_grid = 0
        self.t = 0

    def reward(self):
        return -1*((10*self.state['curr_error'])**2) - 10*(self.state['average_p_grid'] - self.state['agg_load'])**2

    def calc_state(self, obs):
        """
        Provides metrics to evaluate and classify the state of the system.
        :return: dictionary
        """
        current_error = (obs.agg_load - obs.agg_setpoint)
        change_rp = obs.reward_price[0] - obs.reward_price[-1]
        current_rp = obs.reward_price[self.ts]
        time_of_day = obs.timestep % (24 * obs.dt)
        forecast_error = obs.forecast_load[0] - obs.forecast_setpoint
        forecast_trend = obs.forecast_load[0] - obs.forecast_load[-1]
        std_house = np.std(obs.house_load)
        self.t += 1
        self.total_p_grid += obs.agg_load
        self.avg_p_grid = self.total_p_grid / self.t

        state = {"curr_error":current_error,
                "time_of_day":time_of_day,
                "fcst_error":forecast_error,
                "forecast_trend": forecast_trend,
                "delta_action": change_rp,
                "std_house_load": std_house,
                "average_p_grid": self.avg_p_grid,
                "agg_load": obs.agg_load}
        self.state = state
        return state
