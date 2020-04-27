import os

import threading
# import numpy as np
import cvxpy as cp
from redis import StrictRedis

from dragg.mpc_calc_logger import MPCCalcLogger


class MPCCalc(threading.Thread):
    def __init__(self, q, h):
        """

        :param q: queue.Queue
        :param h: int, prediction horizon
        """
        super(MPCCalc, self).__init__()
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
        self.reward_price = None
        self.h_plus = self.horizon + 1
        self.prev_optimal_vals = None  # set after timestep > 0, set_vals_for_current_run

    def write_demand_to_redis(self, d):
        worked = self.redis_client.set(self.home["name"], d)
        if not worked:
            self.mpc_log.logger.error(f"Unable to write to Redis for: {self.home['name']}")

    def redis_write_optimal_vals(self):
        for k, v in self.optimal_vals.items():
            self.redis_client.hset(self.home["name"], k, v)

    def redis_get_prev_optimal_vals(self):
        self.prev_optimal_vals = self.redis_client.hgetall(self.home["name"])

    def setup_base_problem(self):
        if self.timestep == 0:
            self.temp_in_init = cp.Constant(19)
            self.temp_wh_init = cp.Constant(45.5)
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
        self.hvac_cool_on = cp.Variable(self.horizon, boolean=True, name="hvac_cool_on")
        self.hvac_heat_on = cp.Variable(self.horizon, boolean=True, name="hvac_heat_on")
        self.wh_heat_on = cp.Variable(self.horizon, boolean=True, name="wh_heat_on")
        # discomfort_index = cp.Variable(h_plus)

        # Define constants
        self.spp = cp.Constant(self.spp_current)
        self.oat = cp.Constant(self.oat_current)
        self.ghi = cp.Constant(self.ghi_current)

        # Water heater temperature constraints
        self.temp_wh_min = cp.Constant(45.5)
        self.temp_wh_max = cp.Constant(48.5)

        # Home temperature constraints
        self.temp_in_min = cp.Constant(19)
        self.temp_in_max = cp.Constant(21.5)

    def setup_battery_problem(self):
        if self.timestep == 0:
            self.e_batt_init = cp.Constant(0)
        else:
            self.e_batt_init = cp.Constant(float(self.prev_optimal_vals["e_batt_opt"]))

        # Define constants
        self.batt_max_rate = cp.Constant(float(self.home["battery"]["max_rate"]))
        self.batt_cap_total = cp.Constant(float(self.home["battery"]["capacity"]))
        self.batt_cap_min = cp.Constant(float(self.home["battery"]["capacity_lower"]))
        self.batt_cap_max = cp.Constant(float(self.home["battery"]["capacity_upper"]))
        self.batt_ch_eff = cp.Constant(float(self.home["battery"]["ch_eff"]))
        self.batt_disch_eff = cp.Constant(float(self.home["battery"]["disch_eff"]))

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
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
            self.p_grid == self.p_load,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,
            self.temp_wh[1:self.h_plus] <= self.temp_wh_max,
        ]
        # hvac_cool_on <= 1,
        # hvac_cool_on >= 0,
        # wh_heat_on <= 1,
        # wh_heat_on >= 0,
        # temp_in[1:h_plus] == temp_in[0:self.horizon] + 3600 * (((oat[1:h_plus] - temp_in[0:self.horizon]) / home_r) - hvac_cool_on * hvac_p_c * 1000 + hvac_heat_on * hvac_p_h * 1000) / home_c,
        # temp_in[1:h_plus] == temp_in[0:self.horizon] + 3600 * (((oat[1:h_plus] - temp_in[0:self.horizon]) / home_r) - hvac_cool_on * hvac_p_c * 1000 ) / home_c,
        # temp_wh[1:h_plus] == temp_wh[0:self.horizon] + 3600 * (((temp_in[1:h_plus] - temp_wh[0:self.horizon]) / wh_r) + wh_heat_on * wh_p * 1000) / wh_c,
        # p_load == hvac_p_c * hvac_cool_on + wh_p * wh_heat_on,
        # hvac_cool_on * hvac_heat_on == 0,  # either heating or cooling or none, not both.

    def add_battery_constraints(self):
        self.constraints = [
            self.temp_in[0] == self.temp_in_init,
            self.temp_wh[0] == self.temp_wh_init,
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon] + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r) - self.hvac_cool_on * self.hvac_p_c + self.hvac_heat_on * self.hvac_p_h) / self.home_c,
            self.temp_wh[1:self.h_plus] == self.temp_wh[0:self.horizon] + (((self.temp_in[1:self.h_plus] - self.temp_wh[0:self.horizon]) / self.wh_r) + self.wh_heat_on * self.wh_p) / self.wh_c,
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,
            self.temp_wh[1:self.h_plus] <= self.temp_wh_max,
            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,

            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon] + self.batt_ch_eff * self.p_batt_ch[0:self.horizon] / self.batt_cap_total + self.p_batt_disch[0:self.horizon] / (self.batt_disch_eff * self.batt_cap_total),
            self.e_batt[0] == self.e_batt_init,
            self.p_batt_ch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_ch[0:self.horizon] >= 0,
            -self.p_batt_disch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_disch[0:self.horizon] <= 0,
            self.e_batt[1:self.h_plus] <= self.batt_cap_max,
            self.e_batt[1:self.h_plus] >= self.batt_cap_min,
            self.p_load + self.p_batt_ch - self.p_batt_disch >= 0,

            # Change grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch,
            # Do I need constraint that they can't charge/ dischargge at same time
        ]

    def add_pv_constraints(self):
        self.constraints = [
            self.temp_in[0] == self.temp_in_init,
            self.temp_wh[0] == self.temp_wh_init,
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon] + (((self.oat[1:self.h_plus] - self.temp_in[
                0:self.horizon]) / self.home_r) - self.hvac_cool_on * self.hvac_p_c + self.hvac_heat_on * self.hvac_p_h) / self.home_c,
            self.temp_wh[1:self.h_plus] == self.temp_wh[0:self.horizon] + (((self.temp_in[1:self.h_plus] - self.temp_wh[
                0:self.horizon]) / self.wh_r) + self.wh_heat_on * self.wh_p) / self.wh_c,
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,
            self.temp_wh[1:self.h_plus] <= self.temp_wh_max,
            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,

            # PV constraints
            self.p_pv == self.ghi[0:self.horizon] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt),
            self.u_pv_curt >= 0,
            self.u_pv_curt <= 1,

            # Change grid load
            self.p_grid == self.p_load - self.p_pv,
        ]

    def add_battery_pv_constraints(self):
        self.constraints = [
            self.temp_in[0] == self.temp_in_init,
            self.temp_wh[0] == self.temp_wh_init,
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon] + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r) - self.hvac_cool_on * self.hvac_p_c + self.hvac_heat_on * self.hvac_p_h) / self.home_c,
            self.temp_wh[1:self.h_plus] == self.temp_wh[0:self.horizon] + (((self.temp_in[1:self.h_plus] - self.temp_wh[0:self.horizon]) / self.wh_r) + self.wh_heat_on * self.wh_p) / self.wh_c,
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,
            self.temp_wh[1:self.h_plus] <= self.temp_wh_max,
            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,

            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon] + self.batt_ch_eff * self.p_batt_ch[0:self.horizon] / self.batt_cap_total + self.p_batt_disch[0:self.horizon] / (self.batt_disch_eff * self.batt_cap_total),
            self.e_batt[0] == self.e_batt_init,
            self.p_batt_ch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_ch[0:self.horizon] >= 0,
            -self.p_batt_disch[0:self.horizon] <= self.batt_max_rate,
            self.p_batt_disch[0:self.horizon] <= 0,
            self.e_batt[1:self.h_plus] <= self.batt_cap_max,
            self.e_batt[1:self.h_plus] >= self.batt_cap_min,
            self.p_load + self.p_batt_ch - self.p_batt_disch >= 0,

            # PV constraints
            self.p_pv == self.ghi[0:self.horizon] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt),
            self.u_pv_curt >= 0,
            self.u_pv_curt <= 1,

            # Change grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch - self.p_pv,
        ]

    def solve_mpc(self):
        self.obj = cp.Minimize(cp.sum(self.spp[0:self.horizon] * self.p_grid[0:self.horizon]))
        self.prob = cp.Problem(self.obj, self.constraints)
        if not self.prob.is_dcp():
            self.mpc_log.logger.error("Problem is not DCP")
        self.prob.solve(solver=cp.GLPK_MI)

    def cleanup_and_finish(self):
        if self.prob.status in ["infeasible", "unbounded"]:
            self.mpc_log.logger.error(f"Couldn't solve problem for {self.home['name']}: {self.prob.status}")
        elif self.prob.status != "optimal":
            self.mpc_log.logger.info(f"Problem status: {self.prob.status}")
        else:
            self.optimal_vals = {
                "p_grid_opt": self.p_grid.value[0],
                "p_load_opt": self.p_load.value[0],
                "temp_in_opt": self.temp_in.value[1],
                "temp_wh_opt": self.temp_wh.value[1],
                "hvac_cool_on_opt": self.hvac_cool_on.value[0],
                "hvac_heat_on_opt": self.hvac_heat_on.value[0],
                "wh_heat_on_opt": self.wh_heat_on.value[0],
                "cost_opt": self.spp.value[0] * self.p_grid.value[0],
            }
            if 'pv' in self.type:
                self.mpc_log.logger.info("Adding pv optimal vals.")
                self.optimal_vals["p_pv_opt"] = self.p_pv.value[0]
                self.optimal_vals["u_pv_curt_opt"] = self.u_pv_curt.value[0]
            if 'battery' in self.type:
                self.mpc_log.logger.info("Adding battery optimal vals.")
                self.optimal_vals["e_batt_opt"] = self.e_batt.value[1]
                self.optimal_vals["p_batt_ch"] = self.p_batt_ch.value[0]
                self.optimal_vals["p_batt_disch"] = self.p_batt_disch.value[0]
            self.redis_write_optimal_vals()

    def mpc_base(self):
        if self.home["name"] == "Carol-S5FRD":
            self.mpc_log.logger.info(f"Thread: {self.getName()}; home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.add_base_constraints()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_battery(self):
        if self.home["name"] == "David-JONNO":
            self.mpc_log.logger.info(f"Thread: {self.getName()}; home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.setup_battery_problem()
        self.add_battery_constraints()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv(self):
        if self.home["name"] == "Dawn-L23XI":
            self.mpc_log.logger.info(f"Thread: {self.getName()}; home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.setup_pv_problem()
        self.add_pv_constraints()
        self.solve_mpc()
        self.cleanup_and_finish()

    def mpc_pv_battery(self):
        if self.home["name"] == "Myles-XQ5IA":
            self.mpc_log.logger.info(f"Thread: {self.getName()}; home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; SPP: {self.spp_current}")
        self.setup_base_problem()
        self.setup_battery_problem()
        self.setup_pv_problem()
        self.add_battery_pv_constraints()
        self.solve_mpc()
        self.cleanup_and_finish()

    def redis_get_initial_values(self):
        self.start_hour_index = self.redis_client.get('start_hour_index')
        self.current_values = self.redis_client.hgetall("current_values")
        self.all_ghi = self.redis_client.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.lrange('SPP', 0, -1)

    def cast_redis_vals(self):
        self.start_hour_index = int(float(self.start_hour_index))
        self.all_ghi = [int(float(i)) for i in self.all_ghi]
        self.all_oat = [int(float(i)) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]
        self.timestep = int(self.current_values["timestep"])
        self.reward_price = int(self.current_values["reward_price"])
        self.iteration = int(self.current_values["iteration"])
        self.mpc_log.logger.info(f"ts: {self.timestep}")

    def set_vals_for_current_run(self):
        start_slice = self.start_hour_index + self.timestep
        end_slice = start_slice + self.horizon + 1
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.oat_current = self.all_oat[start_slice:end_slice]
        self.spp_current = self.all_spp[start_slice:end_slice]

    def run(self):
        self.redis_get_initial_values()
        self.cast_redis_vals()
        self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()
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
        self.mpc_log.logger.info(f"Queue Empty - thread {self.getName()}")

# def mpc_calc(type, K, tou_rate, t_step, N):
#     """
#     Perform the MPC calculation for a single time period given:
#     :param type: the type of home, one of: [base, battery_only, pv_only, pv_battery]
#     :param K: the planning horizon (in hours)
#     :param tou_rate: the time of use rate, in $/kWh
#     :param t_step: current time step
#     :param N: total number of hours for which calculation will occur
#     """
#
#     if type == "base":
#         mpc_base(K, tou_rate, t_step, N)
#     elif type == "battery_only":
#         mpc_battery(K, tou_rate, t_step, N)
#     elif type == "pv_only":
#         mpc_pv(K, tou_rate, t_step, N)
#     elif type == "pv_battery":
#         mpc_pv_battery(K, tou_rate, t_step, N)
#
#
#
#     # Initialize arrays to store the optimally chosen values for each of the
#     # decision variables at each timestep, N.
#     e_batt_opt = np.zeros(N+1)
#     p_batt_opt = np.zeros(N)
#     p_grid_opt = np.zeros(N)
#     cost = np.zeros(N)
#
#     # Initialize battery condition for t = 0 at 0
#     e_batt_initial = 0
#
#
#     if t_step == 0:
#         e_batt_initial = cp.Constant(0)
#     else:
#         e_batt_initial = cp.Constant(e_batt_opt[t])
#
#     # Define the optimization variables for the problem
#     p_batt = cp.Variable(K)
#     p_grid = cp.Variable(K)
#     e_batt = cp.Variable(K+1)
#
#     # Define constants
#     p_load = cp.Constant(df_load.Load_kWh[t:t+K])  # We are assuming perfect forecasting of load
#     tou = cp.Constant(tou_rate[t:t+K])
#
#     # Define the constraints:
#     constraints = [
#         p_load - p_grid + p_batt == 0,  # ensure that load is met by either the grid or battery
#
#         # Assumptions:
#         e_batt[1:K+1] - e_batt[0:K] - p_batt[0:K] == 0,  # state of charge of battery across timesteps - lossless
#         -p_grid <= 0,  # no power can be pushed back to the grid
#         e_batt[0] == e_batt_initial,  # initial condition of battery at time t
#         p_batt[0:K] <= 2,  # charging rate of battery <= 2kW
#         -p_batt[0:K] <= 2,  # discharge rate of battery <= 2kW
#         e_batt[1:K+1] - 15 <= 0,  # ensure maximum capacity of battery is 15
#         -e_batt[1:K+1] <= 0  # battery charge cannot be negative
#     ]
#
#     # Formulate and solve objective function
#     obj = cp.Minimize(cp.sum(tou * p_grid))
#     prob = cp.Problem(obj, constraints)
#     prob.solve()
#
#     if prob.status not in ["infeasible", "unbounded"]:
#         # Only store the 'chosen' values at time t
#         e_batt_opt[t+1] = e_batt.value[1]
#         p_batt_opt[t] = p_batt.value[0]
#         p_grid_opt[t] = p_grid.value[0]
#         cost[t] = p_grid.value[0] * tou.value[0]
#
#     else:
#         raise ValueError(f"the problem could not be solved.  status: {prob.status}")
#     return e_batt_opt, p_batt_opt, p_grid_opt, cost
