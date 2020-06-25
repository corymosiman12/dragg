import os
import numpy as np
import cvxpy as cp
from redis import StrictRedis
import redis

from dragg.redis_client import RedisClient


class MPCCalc:
    def __init__(self, q, h, dt, discomfort, disutility, case, redis_client, mpc_log):
        """

        :param q: queue.Queue
        :param h: int, prediction horizon
        :param dt: int, number of timesteps per hour
        """
        self.q = q  # Queue
        self.dt = dt
        self.horizon = max(1, h * self.dt)  # Prediction horizon
        self.mpc_log = mpc_log
        self.redis_client = redis_client
        self.home = None  # reset every time home retrieved from Queue
        self.type = None  # reset every time home retrieved from Queue
        self.start_hour_index = None  # set once upon thread init
        self.current_values = None  # set once upon thread init
        self.all_ghi = None  # list, all values in the GHI list, set once upon thread init
        self.all_oat = None  # list, all values in the OAT list, set once upon thread init
        self.all_spp = None  # list, all values in the SPP list, set once upon thread init
        self.home_r = None
        self.home_c = None
        self.hvac_p_c = None
        self.hvac_p_h = None
        self.wh_r = None
        self.wh_c = None
        self.wh_p = None
        self.temp_in_init = None
        self.temp_wh_init = None
        self.p_load = None
        self.temp_in = None
        self.temp_wh = None
        self.p_grid = None
        self.hvac_cool_on = None
        self.hvac_heat_on = None
        self.wh_heat_on = None
        self.spp = None
        self.oat = None
        self.ghi = None
        self.temp_wh_min = None
        self.temp_wh_max = None
        self.temp_in_min = None
        self.temp_in_max = None
        self.optimal_vals = None
        self.iteration = None
        self.timestep = None
        self.h_plus = self.horizon + 1
        self.prev_optimal_vals = None  # set after timestep > 0, set_vals_for_current_run
        self.reward_price = np.zeros(self.horizon)
        self.mode = "run"
        self._discomfort = discomfort
        self._disutility = disutility
        self.case = case

    def redis_write_optimal_vals(self):
        key = self.home["name"]
        for field, value in self.optimal_vals.items():
            self.redis_client.conn.hset(key, field, value)

    def redis_get_prev_optimal_vals(self):
        key = self.home["name"]
        self.prev_optimal_vals = self.redis_client.conn.hgetall(key)

    def setup_base_problem(self, mode="tou"):
        if self.timestep == 0:
            self.temp_in_init = cp.Constant(self.t_in_init)
            self.temp_wh_init = cp.Constant(self.t_wh_init)
        else:
            self.temp_in_init = cp.Constant(float(self.prev_optimal_vals["temp_in_opt"]))
            self.temp_wh_init = cp.Constant(float(self.prev_optimal_vals["temp_wh_opt"]))

        self.home_r = cp.Constant(float(self.home["hvac"]["r"]))
        self.home_c = cp.Constant(float(self.home["hvac"]["c"]))
        self.hvac_p_c = cp.Constant(float(self.home["hvac"]["p_c"]))
        self.hvac_p_h = cp.Constant((float(self.home["hvac"]["p_h"])))
        self.wh_r = cp.Constant(float(self.home["wh"]["r"]))
        self.wh_c = cp.Constant(float(self.home["wh"]["c"]))
        self.wh_p = cp.Constant(float(self.home["wh"]["p"]))

        # Define optimization variables
        self.p_load = cp.Variable(self.horizon, name="p_load")
        self.temp_in = cp.Variable(self.h_plus, name="temp_in")
        self.temp_wh = cp.Variable(self.h_plus, name="temp_wh")
        self.p_grid = cp.Variable(self.horizon, name="p_grid")
        self.hvac_cool_on = cp.Variable(self.horizon, name="hvac_cool_dc")
        self.hvac_heat_on = cp.Variable(self.horizon, name="hvac_heat_dc")
        self.wh_heat_on = cp.Variable(self.horizon, name="wh_heat_dc")

        # Define constants
        # self.spp = cp.Constant(self.spp_current)
        if mode == "tou":
            self.base_price = np.array(self.tou_current, dtype=float)
        elif mode == "spp":
            self.base_price = np.array(self.spp_current, dtype=float)
        self.oat = cp.Constant(self.oat_current)
        self.ghi = cp.Constant(self.ghi_current)

        # tot_price = np.array(self.reward_price) + base_price[:self.horizon]
        # rp_forecast = np.zeros(self.horizon)
        # self.total_price = cp.Constant(np.array(self.reward_price, dtype=float) + base_price[:self.horizon])

        # Water heater temperature constraints
        self.temp_wh_min = cp.Constant(float(self.initial_values["temp_wh_min"]))
        self.temp_wh_max = cp.Constant(float(self.initial_values["temp_wh_max"]))
        self.temp_wh_sp = (self.temp_wh_min + self.temp_wh_max)/2
        self.wf_wh = self.wh_p/2

        # Home temperature constraints
        self.temp_in_min = cp.Constant(float(self.initial_values["temp_in_min"]))
        self.temp_in_max = cp.Constant(float(self.initial_values["temp_in_max"]))
        self.temp_in_sp = cp.Constant((float(self.initial_values["temp_in_min"]) + float(self.initial_values["temp_in_max"])) / 2)
        # set setpoint according to "season"
        self.wf_temp = self.hvac_p_h

        self.p_grid_baseline = cp.Variable(self.horizon, name="p_grid_forecast")

    def setup_battery_problem(self):
        if self.timestep == 0:
            self.e_batt_init = cp.Constant(self.e_b_init)
            self.p_batt_ch_init = cp.Constant(0)
        else:
            self.e_batt_init = cp.Constant(float(self.prev_optimal_vals["e_batt_opt"]))
            self.p_batt_ch_init = cp.Constant(float(self.prev_optimal_vals["p_batt_ch"]) - float(self.prev_optimal_vals["p_batt_disch"]))

        # Define constants
        self.batt_max_rate = cp.Constant(float(self.home["battery"]["max_rate"]))
        self.batt_cap_total = cp.Constant(float(self.home["battery"]["capacity"]))
        self.batt_cap_min = cp.Constant(float(self.home["battery"]["capacity_lower"]))
        self.batt_cap_max = cp.Constant(float(self.home["battery"]["capacity_upper"]))
        self.batt_ch_eff = cp.Constant(float(self.home["battery"]["ch_eff"]))
        self.batt_disch_eff = cp.Constant(float(self.home["battery"]["disch_eff"]))
        self.batt_cons = cp.Constant(float(self.home["battery"]["batt_cons"]))

        # Define battery optimization variables
        self.p_batt_ch = cp.Variable(self.horizon)
        self.p_batt_disch = cp.Variable(self.horizon)
        self.e_batt = cp.Variable(self.h_plus)

    def setup_pv_problem(self):
        # Define constants
        self.pv_area = cp.Constant(float(self.home["pv"]["area"]))
        self.pv_eff = cp.Constant(float(self.home["pv"]["eff"]))

        # Define PV Optimization variables
        self.p_pv = cp.Variable(self.horizon)
        self.u_pv_curt = cp.Variable(self.horizon)

    def add_base_constraints(self):
        self.constraints = [
            self.temp_in[0] == self.temp_in_init,
            self.temp_wh[0] == self.temp_wh_init,
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon] + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / (self.home_r * self.dt)) - self.hvac_cool_on * (self.hvac_p_c / self.dt) + self.hvac_heat_on * (self.hvac_p_h / self.dt)) / (self.home_c),
            self.temp_wh[1:self.h_plus] == self.temp_wh[0:self.horizon] + (((self.temp_in[1:self.h_plus] - self.temp_wh[0:self.horizon]) / (self.wh_r * self.dt)) + self.wh_heat_on * (self.wh_p / self.dt)) / (self.wh_c),
            # self.temp_in[1:self.h_plus] >= self.temp_in_min,
            # self.temp_wh[1:self.h_plus] >= self.temp_wh_min,

            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
            # self.temp_in[1:self.h_plus] <= self.temp_in_max,
            # self.temp_wh[1:self.h_plus] <= self.temp_wh_max,

            self.temp_in[1:self.h_plus] <= 21,
            self.temp_wh[1:self.h_plus] <= 48,

            self.hvac_cool_on <= 1,
            self.hvac_cool_on >= 0,
            self.hvac_heat_on <= 1,
            self.hvac_heat_on >= 0,
            self.wh_heat_on <= 1,
            self.wh_heat_on >= 0,
            self.p_grid >= 0,
        ]

        # set constraints on HVAC by season
        if max(self.oat_current) <= 26: # "winter"
            self.constraints += [self.hvac_cool_on == 0]
            # self.temp_in_sp = cp.Constant(float(self.initial_values["temp_in_min"]))

        if min(self.oat_current) >= 15: # "summer"
            self.constraints += [self.hvac_heat_on == 0]
            # self.temp_in_sp = cp.Constant(float(self.initial_values["temp_in_max"]))

        if self.mode == "baseline" or self.mode == "forecast":
            self.constraints += [self.p_grid_baseline == self.p_grid] # null difference between optimal and forecast
            self.total_price = cp.Constant(np.array(self.base_price[:self.horizon]))
            # self.discomfort = self.discomfort # hard constraints on temp when discomfort is 0 ( @kyri )
            self.discomfort = self._discomfort
            self.disutility = 0.0 # penalizes shift from forecasted baseline
        else: # if self.mode == "run"
            self.constraints += [self.p_grid_baseline == self.baseline_p_grid_opt]
            self.total_price = cp.Constant(np.array(self.reward_price, dtype=float) + self.base_price[:self.horizon])
            if self.case == "rl_agg":
                self.discomfort = 0 # hard constraints on temp when discomfort is 0 ( @kyri ) # uncomment this when responding to an RP signal
                self.disutility = self._disutility # penalizes shift from forecasted baseline
            else:
                self.discomfort = self._discomfort # hard constraints on temp when discomfort is 0 ( @kyri ) # uncomment this for a baseline run
                self.disutility = 0 # penalizes shift from forecasted baseline

    def add_battery_constraints(self):
        self.charge_mag = cp.Variable()
        self.constraints += [
            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon] + self.batt_ch_eff * self.p_batt_ch[0:self.horizon] / self.dt + self.p_batt_disch[0:self.horizon] / self.dt / self.batt_disch_eff,
            self.e_batt[0] == self.e_batt_init,
            self.p_batt_ch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_ch[0:self.horizon] >= 0,
            -self.p_batt_disch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_disch[0:self.horizon] <= 0,
            self.e_batt[1:self.h_plus] <= self.batt_cap_max,
            self.e_batt[1:self.h_plus] >= self.batt_cap_min,
            self.p_load + self.p_batt_ch + self.p_batt_disch >= 0
        ]

    def add_pv_constraints(self):
        self.constraints += [
            # PV constraints.  GHI provided in W/m2 - convert to kWh
            # self.p_pv == self.ghi[0:self.horizon] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt) / 1000, # not sure why this doesn't work
            self.u_pv_curt >= 0,
            self.u_pv_curt <= 1,
        ]
        for i in range(self.horizon):
            self.constraints += [
                self.p_pv[i] == self.ghi[i] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt[i]) / 1000
            ]

    def set_base_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load # p_load = p_hvac + p_wh
        ]
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon] / self.dt) + self.discomfort * (cp.norm(self.temp_in - self.temp_in_sp) + cp.norm(self.temp_wh - self.temp_wh_sp)) + self.disutility * cp.norm(self.p_grid - self.p_grid_baseline))

    def set_battery_only_p_grid(self):
        self.constraints += [
            # Set grid load # try changing wf of batteries and discharge
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch
        ]
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon] / self.dt) + self.discomfort * (self.batt_cons * cp.norm(100*self.e_batt / self.batt_cap_max - 50) + self.wf_temp * cp.norm(self.temp_in - self.temp_in_sp) + self.wf_wh * cp.norm(self.temp_wh - self.temp_wh_sp)) + self.disutility * cp.norm(self.p_grid - self.p_grid_baseline))

    def set_pv_only_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load - self.p_pv
        ]
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon] / self.dt) + self.discomfort * (self.wf_temp * cp.norm(self.temp_in - self.temp_in_sp) + self.wf_wh * cp.norm(self.temp_wh - self.temp_wh_sp)) + self.disutility * cp.norm(self.p_grid - self.p_grid_baseline))

    def set_pv_battery_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch - self.p_pv
        ]
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon] / self.dt) + self.discomfort * (self.batt_cons * cp.norm(100*self.e_batt / self.batt_cap_max - 50) + self.wf_temp * cp.norm(self.temp_in - self.temp_in_sp) + self.wf_wh * cp.norm(self.temp_wh - self.temp_wh_sp)) + self.disutility * cp.norm(self.p_grid - self.p_grid_baseline))

    def solve_mpc(self):
        # self.obj = cp.Minimize(cp.sum((self.total_price) * self.p_grid[0:self.horizon]))
        self.prob = cp.Problem(self.obj, self.constraints)
        # if not self.prob.is_dcp():
        #     self.mpc_log.logger.error("Problem is not DCP")
        self.prob.solve(solver=cp.ECOS, verbose=False)

    def cleanup_and_finish(self):
        if self.prob.status != "optimal":
            self.mpc_log.logger.error(f"Couldn't solve problem for {self.home['name']} of type {self.home['type']}: {self.prob.status}")
        if self.mode == "run":
            # self.mpc_log.logger.info(f"Status for {self.home['name']}: {self.prob.status}")
            self.optimal_vals = {
                "p_grid_opt": self.p_grid.value[0],
                "p_load_opt": self.p_load.value[0],
                "temp_in_opt": self.temp_in.value[1],
                "temp_wh_opt": self.temp_wh.value[1],
                "hvac_cool_on_opt": self.hvac_cool_on.value[0],
                "hvac_heat_on_opt": self.hvac_heat_on.value[0],
                "wh_heat_on_opt": self.wh_heat_on.value[0],
                "cost_opt": (self.total_price.value[0] * self.p_grid.value[0]), # don't get why this was only spp * p_grid -- shoudln't this be some base price + the reward price
            }
            if 'pv' in self.type:
                self.mpc_log.logger.debug("Adding pv optimal vals.")
                self.optimal_vals["p_pv_opt"] = self.p_pv.value[0]
                self.optimal_vals["u_pv_curt_opt"] = self.u_pv_curt.value[0]
            if 'battery' in self.type:
                self.mpc_log.logger.debug("Adding battery optimal vals.")
                self.optimal_vals["e_batt_opt"] = self.e_batt.value[1]
                self.optimal_vals["p_batt_ch"] = self.p_batt_ch.value[0]
                self.optimal_vals["p_batt_disch"] = self.p_batt_disch.value[0]
            self.mpc_log.logger.info(f"{self.home['name']}; Cost {self.prob.value}; p_grid: {self.p_grid.value[0]}; init_temp_in: {self.temp_in.value[0]}; curr_temp_in: {self.temp_in.value[1]}")
            self.redis_write_optimal_vals()
        elif self.mode == "baseline":
            self.baseline_p_grid_opt = self.p_grid.value
        else:
            self.forecast_p_grid = self.p_grid.value

    def mpc_base(self):
        self.setup_base_problem()
        self.add_base_constraints()
        self.set_base_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_battery(self):
        self.setup_base_problem()
        self.setup_battery_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.set_battery_only_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv(self):
        self.setup_base_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_pv_constraints()
        self.set_pv_only_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv_battery(self):
        self.setup_base_problem()
        self.setup_battery_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.add_pv_constraints()
        self.set_pv_battery_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def redis_get_initial_values(self):
        self.start_hour_index = self.redis_client.conn.get('start_hour_index')
        self.initial_values = self.redis_client.conn.hgetall("initial_values")
        self.current_values = self.redis_client.conn.hgetall("current_values")
        self.all_ghi = self.redis_client.conn.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.conn.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.conn.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.conn.lrange('tou', 0, -1)

    def cast_redis_init_vals(self):
        self.start_hour_index = int(float(self.start_hour_index))
        self.t_in_init = float(self.initial_values["temp_in_init"])
        self.t_wh_init = float(self.initial_values["temp_wh_init"])
        self.e_b_init = float(self.initial_values["e_batt_init"])
        self.all_ghi = [float(i) for i in self.all_ghi]
        self.all_oat = [float(i) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]

    def cast_redis_timestep(self):
        self.timestep = int(self.current_values["timestep"])

    def cast_redis_curr_rps(self):
        rp = self.redis_client.conn.lrange('reward_price', 0, -1)
        self.reward_price[:min(len(rp), self.horizon)] = rp[:min(len(rp), self.horizon)]
        try:
            self.iteration = int(self.current_values["iteration"])
        except:
            if self.timestep == 0:
                self.mpc_log.logger.debug("Running a non-iterative aggregator agent. Convergence to maximum allowable load not guarantueed.")
            else:
                pass
        self.mpc_log.logger.info(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def cast_redis_baseline_rps(self):
        self.reward_price = np.zeros(self.horizon)

    def cast_redis_forecast_rps(self, expected_value):
        rp = self.redis_client.conn.lrange('reward_price', 1, -1) # start at the 2nd timestep (roll rps values)
        rp.append(expected_value)
        self.reward_price[:min(len(rp), self.horizon)] = rp[:min(len(rp), self.horizon)]

    def set_vals_for_current_run(self):
        start_slice = self.start_hour_index + self.timestep

        # Need to extend 1 timestep past horizon for OAT slice
        end_slice = start_slice + self.horizon + 1
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.oat_current = self.all_oat[start_slice:end_slice]
        self.spp_current = self.all_spp[start_slice:end_slice]
        self.tou_current = self.all_tou[start_slice:end_slice]

        # self.ghi_current = np.ones(self.horizon+1) @kyri uncomment these for constant environmental values
        # self.oat_current = 10*np.ones(self.horizon+1)
        # self.spp_current = np.ones(self.horizon+1)

    def run(self):
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()
        self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()

            if self.home is None:
                break

            self.mpc_log.logger.debug(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; RP: {self.reward_price}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()

            self.get_baseline()
            self.cast_redis_curr_rps()
            self.mode = "run" # return to mode "run"

            self.type = self.home["type"]
            if self.type == "base":
                self.mpc_base()
            elif self.type == "battery_only":
                self.mpc_battery()
            elif self.type == "pv_only":
                self.mpc_pv()
            elif self.type == "pv_battery":
                self.mpc_pv_battery()
            self.q.task_done()
        self.mpc_log.logger.info(f"Queue Empty.  ts: {self.timestep}; iteration: {self.iteration}; horizon: {self.horizon}")

    def get_baseline(self, forecast_action=0):
        self.cast_redis_baseline_rps()
        self.mode = "baseline"
        self.forecast_rp = forecast_action

        self.mpc_log.logger.debug(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; RP: {self.reward_price}")

        self.type = self.home["type"]
        if self.type == "base":
            self.mpc_base()
        elif self.type == "battery_only":
            self.mpc_battery()
        elif self.type == "pv_only":
            self.mpc_pv()
        elif self.type == "pv_battery":
            self.mpc_pv_battery()

    def forecast(self, expected_value):
        forecasted_values = np.zeros(self.horizon)
        self.mode = "forecast"
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()
        self.cast_redis_forecast_rps(expected_value)
        self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()

            if self.home is None:
                break

            self.mpc_log.logger.debug(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; RP: {self.reward_price}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()
            self.type = self.home["type"]
            if self.type == "base":
                self.mpc_base()
            elif self.type == "battery_only":
                self.mpc_battery()
            elif self.type == "pv_only":
                self.mpc_pv()
            elif self.type == "pv_battery":
                self.mpc_pv_battery()
            self.q.task_done()
            forecasted_values = np.add(self.forecast_p_grid, forecasted_values)
        self.mpc_log.logger.info(f"Queue Empty.  ts: {self.timestep}; iteration: {self.iteration}; horizon: {self.horizon}")
        return forecasted_values[0]
