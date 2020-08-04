import os
import numpy as np
import cvxpy as cp
from redis import StrictRedis
import redis
import scipy.stats

from dragg.redis_client import RedisClient

# from dragg.scratch import TaskQueue, TaskWorker, TaskResultWrapper

class MPCCalc:
    def __init__(self, q, redis_client, mpc_log):
        """
        :param q: queue.Queue
        :param h: int, prediction horizon
        :param dt: int, number of timesteps per hour
        """
        self.q = q  # Queue
        self.log = mpc_log
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
        self.plug_load = None
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
        self.assumed_wh_draw = None
        self.prev_optimal_vals = None  # set after timestep > 0, set_vals_for_current_run

    def redis_write_optimal_vals(self):
        key = self.home["name"]
        for field, value in self.optimal_vals.items():
            self.redis_client.conn.hset(key, field, value)

    def redis_get_prev_optimal_vals(self):
        key = self.home["name"]
        self.prev_optimal_vals = self.redis_client.conn.hgetall(key)

    def setup_base_problem(self):
        self.sub_subhourly_steps = max(1, int(self.home['hems']['sub_subhourly_steps']))
        self.dt = max(1, int(self.home['hems']['hourly_agg_steps'])) * self.sub_subhourly_steps
        self.horizon = max(1, int(self.home['hems']['horizon'])) * self.dt
        self.h_plus = self.horizon + 1
        self.reward_price = np.zeros(self.horizon)
        self.cast_redis_curr_rps()
        self.set_vals_for_current_run()

        self.home_r = cp.Constant(float(self.home["hvac"]["r"]))
        self.home_c = cp.Constant(float(self.home["hvac"]["c"]))
        self.hvac_p_c = cp.Constant(float(self.home["hvac"]["p_c"]))
        self.hvac_p_h = cp.Constant((float(self.home["hvac"]["p_h"])))
        self.wh_r = cp.Constant(float(self.home["wh"]["r"]))
        self.wh_c = cp.Constant(float(self.home["wh"]["c"]))
        self.wh_p = cp.Constant(float(self.home["wh"]["p"]))

        # Define optimization variables
        self.p_load = cp.Variable(self.horizon)
        self.temp_in = cp.Variable(self.h_plus)
        self.temp_wh = cp.Variable(self.h_plus)
        self.p_grid = cp.Variable(self.horizon)
        self.hvac_cool_on = cp.Variable(self.horizon, boolean=True)
        self.hvac_heat_on = cp.Variable(self.horizon, boolean=True)
        self.wh_heat_on = cp.Variable(self.horizon, boolean=True)
        # self.hvac_cool_on = cp.Variable(self.horizon)
        # self.hvac_heat_on = cp.Variable(self.horizon)
        # self.wh_heat_on = cp.Variable(self.horizon)

        # Define constants
        # self.spp = cp.Constant(self.spp_current)
        self.tou_current = np.repeat(self.tou_current, self.sub_subhourly_steps)
        self.oat_current = np.repeat(self.oat_current, self.sub_subhourly_steps)
        self.ghi_current = np.repeat(self.ghi_current, self.sub_subhourly_steps)
        self.base_price = np.array(self.tou_current, dtype=float)
        self.oat = cp.Constant(self.oat_current)
        self.ghi = cp.Constant(self.ghi_current)

        # Water heater temperature constraints
        self.temp_wh_min = float(self.home["wh"]["temp_wh_min"])
        self.temp_wh_max = cp.Constant(float(self.home["wh"]["temp_wh_max"]))
        self.temp_wh_sp = cp.Constant(float(self.home["wh"]["temp_wh_sp"]))
        # self.t_wh_init = float(self.home["wh"]["temp_wh_init"])
        self.t_wh_init = float(self.home["wh"]["temp_wh_min"])
        self.wh_size = cp.Constant(float(self.home["wh"]["tank_size"]))
        self.tap_temp = cp.Constant(12) # assumed cold tap water is about 55 deg F

        # Do water draws with no prior knowledge
        draw_times = np.array(self.home["wh"]["draw_times"])
        draw_size_list = []
        for h in range(self.h_plus):
            t = self.timestep + (h % self.sub_subhourly_steps)
            if t in draw_times:
                ind = np.where(draw_times == t)[0]
                total_draw = sum(np.array(self.home["wh"]["draw_sizes"])[ind])
            else:
                total_draw = 0
            draw_size_list.append(total_draw)
        np.repeat(draw_size_list, self.sub_subhourly_steps)
        self.draw_size = cp.Constant(draw_size_list)
        # self.draw_size = cp.Constant(draw_size_list[0])
        # ed = sum(draw_size_list) / len(draw_size_list)
        # self.expected_draw = cp.Constant(ed)

        # Home temperature constraints
        self.temp_in_min = cp.Constant(float(self.home["hvac"]["temp_in_min"]))
        self.temp_in_max = cp.Constant(float(self.home["hvac"]["temp_in_max"]))
        self.temp_in_sp = cp.Constant(float(self.home["hvac"]["temp_in_sp"]))
        # self.t_in_init = float(self.home["hvac"]["temp_in_init"])
        self.t_in_init = float(self.home["hvac"]["temp_in_min"])

        # set setpoint according to "season"
        self.wf_temp = self.hvac_p_h

        self.p_load_baseline = cp.Variable(self.horizon)

        if self.timestep == 0:
            self.temp_in_init = cp.Constant(self.t_in_init)
            self.temp_wh_init = cp.Constant(self.t_wh_init)
        else:
            self.temp_in_init = cp.Constant(float(self.prev_optimal_vals["temp_in_opt"]))
            self.temp_wh_init = cp.Constant(float(self.prev_optimal_vals["temp_wh_opt"]))

            # percent_draw = draw_size_list[0] / float(self.home["wh"]["tank_size"])
            # self.temp_wh_init = cp.Constant((1-percent_draw) * float(self.prev_optimal_vals["temp_wh_opt"])
            #                                 + percent_draw * 12)

        # add a simplified model of the homes plug load
        # assume ~40% of home energy use is for plug loads
        # assume each home averages ~2.5 kW using HVAC + WH
        # plug load averages 1.5 kW
        # self.plug_load = cp.Constant(np.random.normal(1.5, 0.1, self.horizon))

        # add a predicted water draw at every timestep
        # self.assumed_wh_draw = cp.Constant(np.random.normal(0.3, 0.05, self.horizon)) / 10
        # self.predicted_wh_draw = cp.Constant(np.zeros(self.horizon)) / 4

    def setup_battery_problem(self):
        if self.timestep == 0:
            self.e_batt_init = cp.Constant(float(self.home["battery"]["e_batt_init"]))
            self.p_batt_ch_init = cp.Constant(0)
        else:
            self.e_batt_init = cp.Constant(float(self.prev_optimal_vals["e_batt_opt"]))
            self.p_batt_ch_init = cp.Constant(float(self.prev_optimal_vals["p_batt_ch"])
                                            - float(self.prev_optimal_vals["p_batt_disch"]))

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
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                            + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / (self.home_r * self.dt))
                                            - self.hvac_cool_on * (self.hvac_p_c / self.dt)
                                            + self.hvac_heat_on * (self.hvac_p_h / self.dt)) / (self.home_c),
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,

            self.temp_wh[0] == self.temp_wh_init,
            # self.temp_wh[1:] == (cp.multiply(self.temp_wh[:self.horizon],(self.wh_size - self.draw_size[1:]))/self.wh_size + cp.multiply(self.tap_temp, self.draw_size[1:])/self.wh_size)
            self.temp_wh[1:] == self.temp_wh[:self.horizon]
                                + (((self.temp_in[1:self.h_plus] - self.temp_wh[:self.horizon]) / (self.wh_r * self.dt))
                                + self.wh_heat_on * (self.wh_p / self.dt)) / (self.wh_c),


            self.temp_wh[1:self.h_plus] >= self.temp_wh_min,
            self.temp_wh[1:self.h_plus] <= self.temp_wh_max,

            self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,

            self.hvac_cool_on <= 1,
            self.hvac_cool_on >= 0,
            self.hvac_heat_on <= 1,
            self.hvac_heat_on >= 0,
            self.wh_heat_on <= 1,
            self.wh_heat_on >= 0
        ]

        # set constraints on HVAC by season
        if max(self.oat_current) <= 26: # "winter"
            self.constraints += [self.hvac_cool_on == 0]

        if min(self.oat_current) >= 15: # "summer"
            self.constraints += [self.hvac_heat_on == 0]

        if self.mode == "baseline" or self.mode == "forecast":
            self.constraints += [self.p_load_baseline == self.p_load] # null difference between optimal and forecast
            bp = np.array(self.base_price[:self.horizon])
            self.total_price_values = 100*bp
            self.total_price = cp.Constant(self.total_price_values)

        else: # if self.mode == "run"
            self.baseline_p_load_opt = 0 # change for disutility factor
            self.constraints += [self.p_load_baseline == self.baseline_p_load_opt]
            self.total_price_values = np.array(self.reward_price, dtype=float) + self.base_price[:self.horizon]
            self.total_price = cp.Constant(100*self.total_price_values)

    def add_battery_constraints(self):
        self.charge_mag = cp.Variable()
        self.constraints += [
            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon]
                                        + self.batt_ch_eff * self.p_batt_ch[0:self.horizon] / self.dt
                                        + self.p_batt_disch[0:self.horizon] / self.dt / self.batt_disch_eff,
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
        for i in range(self.horizon): # not sure why this wont set as an array
            self.constraints += [
                self.p_pv[i] == self.ghi[i] * self.pv_area * self.pv_eff * (1 - self.u_pv_curt[i]) / 1000
            ]

    def set_base_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load # p_load = p_hvac + p_wh
        ]

    def set_battery_only_p_grid(self):
        self.constraints += [
            # Set grid load # try changing wf of batteries and discharge
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch
        ]

    def set_pv_only_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load - self.p_pv
        ]

    def set_pv_battery_p_grid(self):
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch - self.p_pv
        ]

    def solve_mpc(self):
        # self.obj = cp.Minimize(cp.sum((self.total_price) * self.p_grid[0:self.horizon]))
        self.cost = cp.Variable(self.horizon)
        # mu = np.zeros(self.horizon)
        # sigma = 0.006 * np.eye(self.horizon) # must be square
        # self.omega = cs.RandomVariableFactory().create_normal_rv(mu, sigma) # random noise for price signal
        # m = 100
        # eta = 0.95 # number of samples and confidence interval
        # if not self.total_price.value.all() == 0:
        #     for i in range(self.horizon):
        #         self.constraints += [
        #             cs.prob(self.cost[i] == ((self.total_price[i] + self.omega) * self.p_grid[i]), m) <= 1 - eta
        #         ]
        # else:
        for i in range(self.horizon):
            self.constraints += [
                self.cost[i] == (self.total_price[i] * self.p_grid[i])
            ]
        self.obj = cp.Minimize(cp.norm(self.cost))
        self.prob = cp.Problem(self.obj, self.constraints)
        if not self.prob.is_dcp():
            self.log.logger.error("Problem is not DCP")
        flag = self.log.logger.getEffectiveLevel() < 20 # outputs from CVX solver if level is debug or lower
        # flag = False
        try:
            self.prob.solve(solver=cp.GUROBI, verbose=flag)
            self.solved = True
        except:
            self.solved = False
        # self.prob.solve(solver=cp.GUROBI, verbose = flag)
        # self.solved = True

    def get_min_hvac_setbacks(self):
        # get maximum(minimum temperature for house)
        new_temp_in_min = cp.Variable()
        new_temp_in_max = cp.Variable()
        obj = cp.Minimize(1) # minimize change to deadband
        cons = [self.temp_in[0] == self.temp_in_init,
                self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                                + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / (self.home_r * self.dt))
                                                - self.hvac_cool_on * (self.hvac_p_c / self.dt)
                                                + self.hvac_heat_on * (self.hvac_p_h / self.dt)) / (self.home_c),
                self.temp_in[1:self.h_plus] >= self.temp_in_min,
                self.temp_in[1:self.h_plus] <= self.temp_in_max,

                self.temp_in[1:self.h_plus] >= new_temp_in_min,
                self.temp_in[1:self.h_plus] <= new_temp_in_max,
                new_temp_in_min <= self.temp_in_min,
                new_temp_in_max >= self.temp_in_max,

                self.hvac_heat_on >= 0,
                self.hvac_heat_on <= 1,
                self.hvac_cool_on >= 0,
                self.hvac_cool_on <= 1,
                self.hvac_heat_on == 1
                ]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=True)
        self.temp_in_min = cp.Constant(new_temp_in_min.value)
        self.temp_in_max = cp.Constant(new_temp_in_max.value)

    def get_min_wh_setbacks(self):
        # get maximum(minimum temperature for water heater)
        new_temp_wh_min = cp.Variable()
        new_temp_wh_max = cp.Variable()
        obj = cp.Minimize(1) # minimize change to deadband
        cons = [self.temp_wh[0] == ((self.wh_size - self.draw_size) * self.temp_wh_init
                            + self.draw_size * self.tap_temp) / self.wh_size,
                self.temp_wh[1] == self.temp_wh[0]
                                    + (((self.temp_in[1] - self.temp_wh[0]) / (self.wh_r * self.dt))
                                    + self.wh_heat_on[0] * (self.wh_p / self.dt)) / (self.wh_c),
                self.temp_wh[2:] == (self.temp_wh[1:self.horizon]*(self.wh_size - self.expected_draw)/self.wh_size + self.tap_temp*self.expected_draw/self.wh_size)
                                    + (((self.temp_in[2:self.h_plus] - self.temp_wh[1:self.horizon]) / (self.wh_r * self.dt))
                                    + self.wh_heat_on[1:] * (self.wh_p / self.dt)) / (self.wh_c),

                self.temp_wh >= new_temp_wh_min,
                self.temp_wh <= new_temp_wh_max,
                new_temp_wh_max >= self.temp_wh_max,
                new_temp_wh_min <= self.temp_wh_min,

                self.wh_heat_on >= 0,
                self.wh_heat_on <= 1,
                self.wh_heat_on == 1
                ]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=True)
        # return [new_temp_wh_min.value, new_temp_wh_max.value]
        self.temp_wh_min = cp.Constant(new_temp_wh_min.value - 2)
        self.temp_wh_max = cp.Constant(new_temp_wh_max.value + 2)

    def error_handler(self):
        new_temp_in_min = cp.Variable()
        new_temp_in_max = cp.Variable()
        new_temp_wh_min = cp.Variable()
        new_temp_wh_max = cp.Variable()
        obj = cp.Minimize(1) # minimize change to deadband
        cons = [self.temp_wh[0] == self.temp_wh_init,
                self.temp_wh[1:] == self.temp_wh[:self.horizon]
                                    + (((self.temp_in[1:self.h_plus] - self.temp_wh[:self.horizon]) / (self.wh_r * self.dt))
                                    + self.wh_heat_on * (self.wh_p / self.dt)) / (self.wh_c),

                self.temp_wh >= new_temp_wh_min,
                self.temp_wh <= new_temp_wh_max,
                new_temp_wh_max >= self.temp_wh_max,
                new_temp_wh_min <= self.temp_wh_min,

                self.wh_heat_on >= 0,
                self.wh_heat_on <= 1,
                self.wh_heat_on == 1,

                self.temp_in[0] == self.temp_in_init,
                self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                                + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / (self.home_r * self.dt))
                                                - self.hvac_cool_on * (self.hvac_p_c / self.dt)
                                                + self.hvac_heat_on * (self.hvac_p_h / self.dt)) / (self.home_c),
                self.temp_in[1:self.h_plus] >= self.temp_in_min,
                self.temp_in[1:self.h_plus] <= self.temp_in_max,

                self.temp_in[1:self.h_plus] >= new_temp_in_min,
                self.temp_in[1:self.h_plus] <= new_temp_in_max,
                new_temp_in_min <= self.temp_in_min,
                new_temp_in_max >= self.temp_in_max,

                self.hvac_heat_on >= 0,
                self.hvac_heat_on <= 1,
                self.hvac_cool_on >= 0,
                self.hvac_cool_on <= 1,
                self.hvac_heat_on == 1,

                self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
                self.p_grid == self.p_load,
                self.cost == (self.total_price * self.p_grid)
        ]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=True)

        self.temp_wh_min = cp.Constant(new_temp_wh_min.value)
        self.temp_wh_max = cp.Constant(new_temp_wh_max.value)
        self.temp_in_min = cp.Constant(new_temp_in_min.value)
        self.temp_in_max = cp.Constant(new_temp_in_max.value)

    def cleanup_and_finish(self):
        n_iterations = 0
        # while self.prob.status != "optimal" and n_iterations < 50:
        while not self.solved and n_iterations < 10:
            self.log.logger.error(f"Couldn't solve problem for {self.home['name']} of type {self.home['type']}: {self.prob.status}")
            self.temp_in_min -= 0.2
            self.temp_wh_min -= 0.2
            self.solve_type_problem()
            n_iterations +=1

        if not self.solved:
            self.error_handler()
            # self.get_min_wh_setbacks()
            # self.get_min_hvac_setbacks()
        # while self.prob.status != "optimal" and self.n_iterations < 1:
        #     self.log.logger.warning((f"Original wh deadband {self.temp_wh_max - self.temp_wh_min} and temp deadband {self.temp_in_max - self.temp_in_min}",
        #                                 f"Problem couldn't be solved for {self.home['name']} of type {self.home['type']} at timestep {self.timestep}: {self.prob.status}."
        #                                 f"Widening temperature deadband for HVAC and water heater."))
        #     self.get_min_hvac_setbacks()
        #     self.get_min_wh_setbacks()
            # self.temp_wh_min = 0
            # self.temp_wh_max = 100
            # self.temp_in_min = 0
            # self.temp_in_max = 100
            # self.constraints += [self.wh_heat_on == 1,
            #                     self.hvac_heat_on == 1]
            # if 'pv' in self.type:
            #     self.constraints += [self.u_pv_curt == 0]
            # self.fake_solve()
            # self.log.logger.warning(f"Changed to wh deadband {self.temp_wh_max - self.temp_wh_min} and temp deadband {self.temp_in_max - self.temp_in_min}")
            # self.n_iterations += 1
            # self.setup_pv_problem()
            # self.add_base_constraints()
            # self.add_pv_constraints()
            # self.set_pv_only_p_grid()
            # self.solve_mpc()
            # self.solve_type_problem()

        if self.mode == "run":
            self.log.logger.info(f"Status for {self.home['name']}: {self.prob.status}")
            self.optimal_vals = {
                "p_grid_opt": np.average(self.p_grid.value[0:self.sub_subhourly_steps]),
                "p_load_opt": np.average(self.p_load.value[0:self.sub_subhourly_steps]),
                "temp_in_opt": self.temp_in.value[self.sub_subhourly_steps+1],
                "temp_wh_opt": self.temp_wh.value[self.sub_subhourly_steps+1],
                "hvac_cool_on_opt": np.average(self.hvac_cool_on.value[0:self.sub_subhourly_steps]),
                "hvac_heat_on_opt": np.average(self.hvac_heat_on.value[0:self.sub_subhourly_steps]),
                "wh_heat_on_opt": np.average(self.wh_heat_on.value[0:self.sub_subhourly_steps]),
                # "cost_opt": (self.total_price.value[0] * self.p_grid.value[0]), # don't get why this was only spp * p_grid -- shoudln't this be some base price + the reward price
                "cost_opt": np.average(self.cost.value[0:self.sub_subhourly_steps])
            }
            # self.optimal_vals = {
            #     "p_grid_opt": self.p_grid[0].value,
            #     "p_load_opt": self.p_load[0].value,
            #     "temp_in_opt": self.temp_in[1].value,
            #     "temp_wh_opt": self.temp_wh[1].value,
            #     "hvac_cool_on_opt": self.hvac_cool_on[0].value,
            #     "hvac_heat_on_opt": self.hvac_heat_on[0].value,
            #     "wh_heat_on_opt": self.wh_heat_on[0].value,
            #     # "cost_opt": (self.total_price.value[0] * self.p_grid.value[0]), # don't get why this was only spp * p_grid -- shoudln't this be some base price + the reward price
            #     "cost_opt": self.cost[0].value
            # }
            if 'pv' in self.type:
                self.log.logger.debug("Adding pv optimal vals.")
                self.optimal_vals["p_pv_opt"] = np.average(self.p_pv.value[0:self.sub_subhourly_steps])
                self.optimal_vals["u_pv_curt_opt"] = np.average(self.u_pv_curt.value[0:self.sub_subhourly_steps])
            if 'battery' in self.type:
                self.log.logger.debug("Adding battery optimal vals.")
                self.optimal_vals["e_batt_opt"] = self.e_batt.value[self.sub_subhourly_steps+1]
                self.optimal_vals["p_batt_ch"] = np.average(self.p_batt_ch.value[0:self.sub_subhourly_steps])
                self.optimal_vals["p_batt_disch"] = np.average(self.p_batt_disch.value[0:self.sub_subhourly_steps])
            self.log.logger.info(f"MPC solved with status {self.prob.status} for {self.home['name']}; Cost {self.prob.value}; p_grid: {self.p_grid.value[0]}; init_temp_in: {self.temp_in.value[0]}; curr_temp_in: {self.temp_in.value[1]}")
            self.redis_write_optimal_vals()
        elif self.mode == "baseline":
            self.baseline_p_load_opt = self.p_load.value
            if 'battery' in self.type:
                self.baseline_e_batt_opt = self.e_batt.value

        else:
            self.x = np.average(self.p_grid.value[0:self.sub_subhourly_steps])

    def mpc_base(self):
        # self.setup_base_problem()
        self.add_base_constraints()
        self.set_base_p_grid()
        self.solve_mpc()
        # self.cleanup_and_finish()

    def mpc_battery(self):
        # self.setup_base_problem()
        self.setup_battery_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.set_battery_only_p_grid()
        self.solve_mpc()
        # self.cleanup_and_finish()

    def mpc_pv(self):
        # self.setup_base_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_pv_constraints()
        self.set_pv_only_p_grid()
        self.solve_mpc()
        # self.cleanup_and_finish()

    def mpc_pv_battery(self):
        # self.setup_base_problem()
        self.setup_battery_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.add_pv_constraints()
        self.set_pv_battery_p_grid()
        self.solve_mpc()
        # self.cleanup_and_finish()

    def redis_get_initial_values(self):
        self.start_hour_index = self.redis_client.conn.get('start_hour_index')
        # self.initial_values = self.redis_client.conn.hgetall("initial_values")
        self.current_values = self.redis_client.conn.hgetall("current_values")
        self.all_ghi = self.redis_client.conn.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.conn.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.conn.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.conn.lrange('tou', 0, -1)

    def cast_redis_init_vals(self):
        self.start_hour_index = int(float(self.start_hour_index))
        # self.t_in_init = float(self.initial_values["temp_in_init"])
        # self.t_wh_init = float(self.initial_values["temp_wh_init"])
        # self.e_b_init = float(self.initial_values["e_batt_init"])
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
                self.log.logger.debug("Running a non-iterative aggregator agent. Convergence to maximum allowable load not guarantueed.")
            else:
                pass
        self.log.logger.info(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def cast_redis_baseline_rps(self):
        self.reward_price = np.zeros(self.horizon)

    # def cast_redis_forecast_rps(self, expected_value):
    #     rp = self.redis_client.conn.lrange('reward_price', 1, -1) # start at the 2nd timestep (roll rps values)
    #     rp.append(expected_value)
    #     self.reward_price[:min(len(rp), self.horizon)] = rp[:min(len(rp), self.horizon)]

    def set_vals_for_current_run(self):
        start_slice = self.start_hour_index + self.timestep

        # Need to extend 1 timestep past horizon for OAT slice
        end_slice = start_slice + (self.horizon // self.sub_subhourly_steps) + 1
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.oat_current = self.all_oat[start_slice:end_slice]
        # self.spp_current = self.all_spp[start_slice:end_slice]
        self.tou_current = self.all_tou[start_slice:end_slice]

    def solve_type_problem(self):
        self.type = self.home["type"]
        self.type = self.home["type"]
        if self.type == "base":
            self.mpc_base()
        elif self.type == "battery_only":
            self.mpc_battery()
        elif self.type == "pv_only":
            self.mpc_pv()
        elif self.type == "pv_battery":
            self.mpc_pv_battery()

    def run_threaded(self, home):
        self.mode = "run" # return to mode "run"
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()
        # self.set_vals_for_current_run()
        self.home = home

        if not self.home is None:
            # self.log.logger.debug(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; RP: {self.reward_price}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()

            # self.cast_redis_curr_rps()

            self.n_iterations = 0 # iteration counter for solving home MPC problem
            self.setup_base_problem()
            self.solve_type_problem()
            self.cleanup_and_finish()

            # self.q.task_done()

    def run(self):
        # while not self.q.empty():
        #     print("hello")
        #     self.q.task_done()
        self.mode = "run" # return to mode "run"
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()
        # self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()

            if self.home is None:
                break

            self.log.logger.debug(f"Home: {self.home['name']}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()

            # self.cast_redis_curr_rps()

            self.n_iterations = 0 # iteration counter for solving home MPC problem
            self.setup_base_problem()
            self.solve_type_problem()
            self.cleanup_and_finish()

            self.q.task_done()
        self.log.logger.info(f"Queue Empty.  ts: {self.timestep}; iteration: {self.iteration}; horizon: {self.horizon}")

    def forecast(self, expected_value):
        forecasted_values = [0]
        self.mode = "forecast"
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()
        # self.cast_redis_curr_rps(expected_value)
        # self.set_vals_for_current_run()
        while not self.q.empty():
            self.home = self.q.get()

            if self.home is None:
                break

            # self.log.logger.debug(f"Home: {self.home['name']}; ts: {self.timestep}; iter: {self.iteration}; GHI: {self.ghi_current}; OAT: {self.oat_current}; RP: {self.reward_price}")
            if self.timestep > 0:
                self.redis_get_prev_optimal_vals()

            self.n_iterations = 0
            self.setup_base_problem()
            self.solve_type_problem()
            self.cleanup_and_finish()

            self.q.task_done()
            # forecasted_values = np.add(self.forecast_p_grid, forecasted_values)
            forecasted_values[0] += self.x
        self.log.logger.info(f"Queue Empty.  ts: {self.timestep}; iteration: {self.iteration}; horizon: {self.horizon}")
        return forecasted_values[0]
