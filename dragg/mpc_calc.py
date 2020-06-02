import os

import numpy as np
import cvxpy as cp
from redis import StrictRedis

from dragg.mpc_calc_logger import MPCCalcLogger


class MPCCalc:
    def __init__(self, q, h):
        """

        :param q: queue.Queue
        :param h: int, prediction horizon
        """
        self.q = q  # Queue
        self.horizon = h  # Prediction horizon
        self.mpc_log = MPCCalcLogger()
        self.redis_client = StrictRedis(host=os.environ.get('REDIS_HOST', 'localhost'), decode_responses=True)
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

    def redis_write_optimal_vals(self):
        for k, v in self.optimal_vals.items():
            self.redis_client.hset(self.home["name"], k, v)

    def redis_get_prev_optimal_vals(self):
        self.prev_optimal_vals = self.redis_client.hgetall(self.home["name"])

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
            base_price = np.array(self.tou_current, dtype=float)
        elif mode == "spp":
            base_price = np.array(self.spp_current, dtype=float)
        self.oat = cp.Constant(self.oat_current)
        self.ghi = cp.Constant(self.ghi_current)

        # tot_price = np.array(self.reward_price) + base_price[:self.horizon]
        # rp_forecast = np.zeros(self.horizon)
        self.total_price = cp.Constant(np.array(self.reward_price, dtype=float) + base_price[:self.horizon])

        # Water heater temperature constraints
        self.temp_wh_min = cp.Constant(float(self.initial_values["temp_wh_min"]))
        self.temp_wh_max = cp.Constant(float(self.initial_values["temp_wh_max"]))
        self.temp_wh_sp = (self.temp_wh_min + self.temp_wh_max)/2
        self.wf_wh = self.wh_p/2

        # Home temperature constraints
        self.temp_in_min = cp.Constant(float(self.initial_values["temp_in_min"]))
        self.temp_in_max = cp.Constant(float(self.initial_values["temp_in_max"]))
        # set setpoint according to "season"
        self.wf_temp = self.hvac_p_h

        self.discomfort = self.total_price[0]

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
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon] + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r) - self.hvac_cool_on * self.hvac_p_c + self.hvac_heat_on * self.hvac_p_h) / self.home_c,
            self.temp_wh[1:self.h_plus] == self.temp_wh[0:self.horizon] + (((self.temp_in[1:self.h_plus] - self.temp_wh[0:self.horizon]) / self.wh_r) + self.wh_heat_on * self.wh_p) / self.wh_c,
            # self.temp_in[1:self.h_plus] >= self.temp_in_min,
            # self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
            # self.temp_in[1:self.h_plus] <= self.temp_in_max,
            # self.temp_wh[1:self.h_plus] <= self.temp_wh_max,
            self.hvac_cool_on <= 1,
            self.hvac_cool_on >= 0,
            self.hvac_heat_on <= 1,
            self.hvac_heat_on >= 0,
            self.wh_heat_on <= 1,
            self.wh_heat_on >= 0,
            self.p_grid >= 0
        ]

        if max(self.oat_current) <= 26:
            self.constraints += [self.hvac_cool_on == 0]
            self.temp_in_sp = cp.Constant(float(self.initial_values["temp_in_min"]))

        if min(self.oat_current) >= 15:
            self.constraints += [self.hvac_heat_on == 0]
            self.temp_in_sp = cp.Constant(float(self.initial_values["temp_in_max"]))

    def add_battery_constraints(self):
        self.constraints += [
            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon] + self.batt_ch_eff * self.p_batt_ch[0:self.horizon] + self.p_batt_disch[0:self.horizon] / self.batt_disch_eff,
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
            # PV constraints.  GHI provided in W/m2 - convert to kW
            self.p_pv == self.ghi[0:self.horizon] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt) / 1000,
            self.u_pv_curt >= 0,
            self.u_pv_curt <= 1,
        ]

    def set_base_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load
        ]
        self.wf_wh = 1.4 * self.wf_wh
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon]) + self.discomfort * (self.wf_temp * cp.sum(cp.norm(self.temp_in - self.temp_in_sp)) + self.wf_wh * cp.sum(cp.norm(self.temp_wh - self.temp_wh_sp))))


    def set_battery_only_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch
        ]
        self.wf_wh = 2 * self.wf_wh
        self.batt_cons = 0.5
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon]) + self.discomfort * (self.batt_cons * cp.sum(cp.norm(self.e_batt / self.batt_cap_total - 0.5)) + self.wf_temp * cp.sum(cp.norm(self.temp_in - self.temp_in_sp)) + self.wf_wh * cp.sum(cp.norm(self.temp_wh - self.temp_wh_sp))))

    def set_pv_only_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load - self.p_pv
        ]
        self.wf_temp = self.hvac_p_h*1.5
        self.wf_wh = self.wh_p*2
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon]) + self.discomfort * (self.wf_temp * cp.sum(cp.norm(self.temp_in - self.temp_in_sp)) + self.wf_wh * cp.sum(cp.norm(self.temp_wh - self.temp_wh_sp))))

    def set_pv_battery_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch - self.p_pv
        ]
        self.wf_wh = 1.5 * self.wf_wh
        self.obj = cp.Minimize(cp.sum(self.total_price * self.p_grid[0:self.horizon]) + self.discomfort * (self.batt_cons * cp.sum(cp.norm(self.e_batt / self.batt_cap_total - 0.5)) + self.wf_temp * cp.sum(cp.norm(self.temp_in - self.temp_in_sp)) + self.wf_wh * cp.sum(cp.norm(self.temp_wh - self.temp_wh_sp)))) # i think this was previously adding the current negotiated reward price to every timestep

    def solve_mpc(self):
        # self.obj = cp.Minimize(cp.sum((self.total_price) * self.p_grid[0:self.horizon]))
        self.prob = cp.Problem(self.obj, self.constraints)
        # if not self.prob.is_dcp():
        #     self.mpc_log.logger.error("Problem is not DCP")
        self.prob.solve(solver=cp.ECOS, verbose=False)

    def cleanup_and_finish(self):
        # if self.prob.status in ["infeasible", "unbounded"]:
        if self.prob.status != "optimal":
            self.mpc_log.logger.error(f"Couldn't solve problem for {self.home['name']}: {self.prob.status}")
        # elif self.prob.status != "optimal":
        #     self.mpc_log.logger.info(f"Problem status: {self.prob.status}")
        else:
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
                # self.mpc_log.logger.info("Adding pv optimal vals.")
                self.optimal_vals["p_pv_opt"] = self.p_pv.value[0]
                self.optimal_vals["u_pv_curt_opt"] = self.u_pv_curt.value[0]
            if 'battery' in self.type:
                # self.mpc_log.logger.info("Adding battery optimal vals.")
                self.optimal_vals["e_batt_opt"] = self.e_batt.value[1]
                self.optimal_vals["p_batt_ch"] = self.p_batt_ch.value[0]
                self.optimal_vals["p_batt_disch"] = self.p_batt_disch.value[0]
            self.mpc_log.logger.info(f"{self.home['name']}; Cost {self.prob.value}; p_grid: {self.p_grid.value[0]}; temp_in: {self.temp_in.value[1]}")
            self.redis_write_optimal_vals()

    def mpc_base(self):
        # Sanity check on one home
        # self.mpc_log.logger.info(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.add_base_constraints()
        self.set_base_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_battery(self):
        # Sanity check on one home
        # if self.home["name"] == "David-JONNO":
        # self.mpc_log.logger.info(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.setup_battery_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.set_battery_only_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv(self):
        # Sanity check on one home
        # if self.home["name"] == "Dawn-L23XI":
        #     self.mpc_log.logger.info(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_pv_constraints()
        self.set_pv_only_p_grid()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv_battery(self):
        # Sanity check on one home
        # if self.home["name"] == "Myles-XQ5IA":
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
        self.start_hour_index = self.redis_client.get('start_hour_index')
        self.initial_values = self.redis_client.hgetall("initial_values")
        self.current_values = self.redis_client.hgetall("current_values")
        self.all_ghi = self.redis_client.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.lrange('tou', 0, -1)

    def cast_redis_vals(self):
        self.start_hour_index = int(float(self.start_hour_index))
        self.t_in_init = float(self.initial_values["temp_in_init"])
        self.t_wh_init = float(self.initial_values["temp_wh_init"])
        self.e_b_init = float(self.initial_values["e_batt_init"])
        self.min_runtime = int(float(self.initial_values["min_runtime_mins"]))
        self.min_runtime_fraction = self.min_runtime / 60  # proportion of an hour
        self.min_runtime_fraction_inv = 1 / self.min_runtime_fraction
        self.all_ghi = [float(i) for i in self.all_ghi]
        self.all_oat = [float(i) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]
        self.timestep = int(self.current_values["timestep"])
        # self.current_rp = float(self.current_values["reward_price"])
        self.reward_price = self.redis_client.lrange('reward_price', 0, -1)
        try:
            self.iteration = int(self.current_values["iteration"])
        except:
            if self.timestep == 0:
                self.mpc_log.logger.debug("Running a non-iterative aggregator agent. Convergence not guarantueed.")
            else:
                pass
        self.mpc_log.logger.info(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def set_vals_for_current_run(self):
        start_slice = self.start_hour_index + self.timestep

        # Need to extend 1 hr past horizon for OAT slice
        end_slice = start_slice + self.horizon + 1
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.oat_current = self.all_oat[start_slice:end_slice]
        self.spp_current = self.all_spp[start_slice:end_slice]
        self.tou_current = self.all_tou[start_slice:end_slice]

    def run(self):
        self.redis_get_initial_values()
        self.cast_redis_vals()
        self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()
            self.mpc_log.logger.info(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()
            if self.home is None:
                break
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
