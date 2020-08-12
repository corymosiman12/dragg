import os
import numpy as np
import cvxpy as cp
from redis import StrictRedis
import redis
import scipy.stats

from dragg.redis_client import RedisClient

class MPCCalc:
    def __init__(self, redis_client):
        """
        :param redis_client: redis.Redis
        """
        self.q = None # depricated for threaded uses
        self.log = None # depricated for threaded uses, set with house
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
        """
        Sends the optimal values for each home to the redis server.
        :return: None
        """
        # print("writing opt vals")
        key = self.home["name"]
        for field, value in self.optimal_vals.items():
            self.redis_client.conn.hset(key, field, value)

    def redis_get_prev_optimal_vals(self):
        """
        Collects starting point environmental values for all homes (such as current temperature).
        :return: None
        """
        key = self.home["name"]
        self.prev_optimal_vals = self.redis_client.conn.hgetall(key)

    def setup_base_problem(self):
        """
        Sets variable objects for CVX optimization problem. Includes "base home"
        systems of HVAC and water heater and other environmental values.
        :return: None
        """
        # print("creating base")
        self.sub_subhourly_steps = max(1, int(self.home['hems']['sub_subhourly_steps']))
        self.dt = max(1, int(self.home['hems']['hourly_agg_steps'])) * self.sub_subhourly_steps
        self.horizon = max(1, int(self.home['hems']['horizon'] * self.dt))
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

        # Define constants
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
        self.t_wh_init = float(self.home["wh"]["temp_wh_init"])
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
        self.t_in_init = float(self.home["hvac"]["temp_in_init"])

        if self.timestep == 0:
            self.temp_in_init = cp.Constant(self.t_in_init)
            self.temp_wh_init = cp.Constant(self.t_wh_init)
        else:
            self.temp_in_init = cp.Constant(float(self.prev_optimal_vals["temp_in_opt"]))
            self.temp_wh_init = cp.Constant(float(self.prev_optimal_vals["temp_wh_opt"]))

            # percent_draw = draw_size_list[0] / float(self.home["wh"]["tank_size"])
            # self.temp_wh_init = cp.Constant((1-percent_draw) * float(self.prev_optimal_vals["temp_wh_opt"])
            #                                 + percent_draw * 12)
        # print("base created")

    def setup_battery_problem(self):
        """
        Adds CVX variables for battery subsystem in battery and battery_pv homes.
        :return: None
        """
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
        """
        Adds CVX variables for photovoltaic subsystem in pv and battery_pv homes.
        :return: None
        """
        # Define constants
        self.pv_area = cp.Constant(float(self.home["pv"]["area"]))
        self.pv_eff = cp.Constant(float(self.home["pv"]["eff"]))

        # Define PV Optimization variables
        self.p_pv = cp.Variable(self.horizon)
        self.u_pv_curt = cp.Variable(self.horizon)

    def add_base_constraints(self):
        """
        Creates the system dynamics for thermal energy storage systems: HVAC and
        water heater.
        :return: None
        """
        self.constraints = [
            # Indoor air temperature constraints
            self.temp_in[0] == self.temp_in_init,
            self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                            + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r)
                                            - self.hvac_cool_on * self.hvac_p_c
                                            + self.hvac_heat_on * self.hvac_p_h) / (self.home_c * self.dt),
            self.temp_in[1:self.h_plus] >= self.temp_in_min,
            self.temp_in[1:self.h_plus] <= self.temp_in_max,

            # Hot water heater contraints
            self.temp_wh[0] == self.temp_wh_init,
            # self.temp_wh[1:] == (cp.multiply(self.temp_wh[:self.horizon],(self.wh_size - self.draw_size[1:]))/self.wh_size + cp.multiply(self.tap_temp, self.draw_size[1:])/self.wh_size)
            self.temp_wh[1:] == self.temp_wh[:self.horizon]
                                + (((self.temp_in[1:self.h_plus] - self.temp_wh[:self.horizon]) / self.wh_r)
                                + self.wh_heat_on * self.wh_p) / (self.wh_c * self.dt),

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

        # Set constraints on HVAC by season
        if max(self.oat_current) <= 26: # "winter"
            self.constraints += [self.hvac_cool_on == 0]

        if min(self.oat_current) >= 15: # "summer"
            self.constraints += [self.hvac_heat_on == 0]

        # set total price for electricity
        self.total_price_values = np.array(self.reward_price, dtype=float) + self.base_price[:self.horizon]
        self.total_price = cp.Constant(100*self.total_price_values)

    def add_battery_constraints(self):
        """
        Creates the system dynamics for chemical energy storage.
        :return: None
        """
        self.charge_mag = cp.Variable()
        self.constraints += [
            # Battery constraints
            self.e_batt[1:self.h_plus] == self.e_batt[0:self.horizon]
                                        + (self.batt_ch_eff * self.p_batt_ch[0:self.horizon]
                                        + self.p_batt_disch[0:self.horizon] / self.batt_disch_eff) / self.dt,
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
        """
        Creates the system dynamics for photovoltaic generation. (Using measured GHI.)
        :return: None
        """
        self.constraints += [
            # PV constraints.  GHI provided in W/m2 - convert to kWh
            self.p_pv == self.pv_area * self.pv_eff * cp.multiply(self.ghi[0:self.horizon], (1 - self.u_pv_curt)) / 1000,
            self.u_pv_curt >= 0,
            self.u_pv_curt <= 1,
        ]

    def set_base_p_grid(self):
        """
        Sets p_grid of home to equal the load of the HVAC and water heater. To
        be used if and only if home type is base.
        :return: None
        """
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load # p_load = p_hvac + p_wh
        ]

    def set_battery_only_p_grid(self):
        """
        Sets p_grid of home to equal the load of the HVAC and water heater, plus
        or minus the charge/discharge of the battery. To be used if and only if
        home is of type battery_only.
        :return: None
        """
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch
        ]

    def set_pv_only_p_grid(self):
        """
        Sets p_grid of home equal to the load of the HVAC and water heater minus
        potential generation from the PV subsystem. To be used if and only if the
        home is of type pv_only.
        :return: None
        """
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load - self.p_pv
        ]

    def set_pv_battery_p_grid(self):
        """
        Sets p_grid of home equal to the load of the HVAC and water heater, plus
        or minus the charge/discharge of the battery, minus potential generation
        from the PV subsystem. To be used if and only if the home is of type pv_battery.
        :return: None
        """
        self.constraints += [
            # Set grid load
            self.p_grid == self.p_load + self.p_batt_ch + self.p_batt_disch - self.p_pv
        ]

    def solve_mpc(self):
        """
        Sets the objective function of the Home Energy Management System to be the
        minimization of cost over the MPC time horizon and solves via CVXPY.
        Used for all home types.
        :return: None
        """
        # print("solving")
        self.cost = cp.Variable(self.horizon)
        self.constraints += [self.cost == cp.multiply(self.total_price, self.p_grid)] # think this should work
        self.obj = cp.Minimize(cp.norm(self.cost))
        self.prob = cp.Problem(self.obj, self.constraints)
        if not self.prob.is_dcp():
            # self.log.logger.error("Problem is not DCP")
            print("prob is not dcp")
        # flag = self.log.logger.getEffectiveLevel() < 20 # outputs from CVX solver if level is debug or lower
        flag = False
        try:
            self.prob.solve(solver=cp.GUROBI, verbose=flag)
            self.solved = True
        except:
            self.solved = False

    def get_min_hvac_setbacks(self):
        """
        Solves for the minimimum change required by the HEMS HVAC deadband in
        order to make the optimization feasible.
        Resets the HEMS constraints with the new temperature bounds.
        :return: None
        """
        # self.log.logger.info("Setting minimum changes to HVAC.")
        new_temp_in_min = cp.Variable()
        new_temp_in_max = cp.Variable()
        obj = cp.Minimize(new_temp_in_max - new_temp_in_min) # minimize change to deadband
        cons = [self.temp_in[0] == self.temp_in_init,
                self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                                + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r)
                                                - self.hvac_cool_on * self.hvac_p_c
                                                + self.hvac_heat_on * self.hvac_p_h) / (self.home_c * self.dt),

                self.temp_in[1:self.h_plus] >= new_temp_in_min,
                self.temp_in[1:self.h_plus] <= new_temp_in_max,
                new_temp_in_min <= self.temp_in_min,
                new_temp_in_max >= self.temp_in_max,

                self.hvac_heat_on <= 1,
                self.hvac_heat_on >= 0,
                self.hvac_cool_on <= 1,
                self.hvac_cool_on >= 0,
                ]

        # set constraints on HVAC by season
        if max(self.oat_current) <= 26: # "winter"
            cons += [self.hvac_cool_on == 0]

        if min(self.oat_current) >= 15: # "summer"
            cons += [self.hvac_heat_on == 0]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=False)
        self.temp_in_min = cp.Constant(new_temp_in_min.value)
        self.temp_in_max = cp.Constant(new_temp_in_max.value)
        self.hvac_cool_on = cp.Constant(self.hvac_cool_on.value)
        self.hvac_heat_on = cp.Constant(self.hvac_heat_on.value)
        # print(self.temp_in.value)

    def get_min_wh_setbacks(self):
        """
        Solves for the minimimum change required by the HEMS water heater deadband in
        order to make the optimization feasible.
        Resets the HEMS constraints with the new temperature bounds.
        :return: None
        """
        # self.log.logger.info("Setting minimum changes to water heater.")
        new_temp_wh_min = cp.Variable()
        new_temp_wh_max = cp.Variable()
        obj = cp.Minimize(cp.abs(self.temp_wh_min - new_temp_wh_min) + cp.abs(self.temp_wh_max - new_temp_wh_max)) # minimize change to deadband
        cons = [self.temp_wh[0] == self.temp_wh_init,
                # self.temp_wh[1:] == (cp.multiply(self.temp_wh[:self.horizon],(self.wh_size - self.draw_size[1:]))/self.wh_size + cp.multiply(self.tap_temp, self.draw_size[1:])/self.wh_size)
                self.temp_wh[1:] == self.temp_wh[:self.horizon]
                                    + (((self.temp_in[1:self.h_plus] - self.temp_wh[:self.horizon]) / self.wh_r)
                                    + self.wh_heat_on * self.wh_p) / (self.wh_c * self.dt),

                self.temp_wh >= new_temp_wh_min,
                self.temp_wh <= new_temp_wh_max,
                new_temp_wh_max >= self.temp_wh_max,
                new_temp_wh_min <= self.temp_wh_min,

                self.wh_heat_on >= 0,
                self.wh_heat_on <= 1
                ]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=False)
        # self.log.logger.info(f"Problem status {prob.status}")
        self.temp_wh_min = cp.Constant(new_temp_wh_min.value)
        self.temp_wh_max = cp.Constant(new_temp_wh_max.value)
        self.wh_heat_on = cp.Constant(self.wh_heat_on.value)

    def get_alt_p_grid(self):
        # self.log.logger.info("Setting p_grid using alternate objectives.")
        self.constraints = [self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
                            self.cost == cp.multiply(self.total_price, self.p_grid)] # reset constraints

        if self.type=="pv_battery":
            self.set_pv_battery_p_grid()
        elif self.type=="battery_only":
            self.set_battery_only_p_grid()
        elif self.type=="pv_only":
            self.set_pv_only_p_grid()
        else:
            self.set_base_p_grid()
        self.obj = cp.Minimize(cp.multiply(self.total_price, self.p_grid))
        self.prob = cp.Problem(self.obj, self.constraints)
        self.prob.solve(solver=cp.GUROBI, verbose=False)
        # self.log.logger.info(f"Problem status {self.prob.status}")

    def error_handler(self):
        """
        Utilizes the CVXPY problem as setup previously to brute force a control signal.
        Turns on HVAC (cooling or heating) and water heater for all timesteps.
        Not intended to consider costs, only to resolve errors in normal solve process.
        :return:
        """
        new_temp_in_min = cp.Variable()
        new_temp_in_max = cp.Variable()
        new_temp_wh_min = cp.Variable()
        new_temp_wh_max = cp.Variable()
        obj = cp.Maximize(cp.sum(self.hvac_heat_on) + cp.sum(self.hvac_cool_on))
        cons = [self.temp_wh[0] == self.temp_wh_init,
                self.temp_wh[1:] == self.temp_wh[:self.horizon]
                                    + (((self.temp_in[1:self.h_plus] - self.temp_wh[:self.horizon]) / self.wh_r)
                                    + self.wh_heat_on * self.wh_p) / (self.wh_c * self.dt),

                self.temp_wh >= new_temp_wh_min,
                self.temp_wh <= new_temp_wh_max,
                new_temp_wh_max >= self.temp_wh_max,
                new_temp_wh_min <= self.temp_wh_min,

                self.wh_heat_on >= 0,
                self.wh_heat_on <= 1,
                self.wh_heat_on == 1,

                self.temp_in[0] == self.temp_in_init,
                self.temp_in[1:self.h_plus] == self.temp_in[0:self.horizon]
                                                + (((self.oat[1:self.h_plus] - self.temp_in[0:self.horizon]) / self.home_r)
                                                - self.hvac_cool_on * self.hvac_p_c
                                                + self.hvac_heat_on * self.hvac_p_h) / (self.home_c * self.dt),
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

                self.p_load == self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on,
                self.p_grid == self.p_load,
                self.cost == cp.multiply(self.total_price, self.p_grid)
        ]

        if 'battery' in self.type:
            self.constraints += [self.p_batt_ch == 0,
                                 self.p_batt_disch == 0]

        if 'pv' in self.type:
            self.constraints += [self.u_pv_curt == 0]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GUROBI, verbose=False)

        self.temp_wh_min = cp.Constant(new_temp_wh_min.value)
        self.temp_wh_max = cp.Constant(new_temp_wh_max.value)
        self.temp_in_min = cp.Constant(new_temp_in_min.value)
        self.temp_in_max = cp.Constant(new_temp_in_max.value)

    def cleanup_and_finish(self):
        """
        Resolves .solve_mpc() with error handling and collects all data on solver.
        :return: None
        """
        n_iterations = 0
        while not self.solved and n_iterations < 10:
            # self.log.logger.error(f"Couldn't solve problem for {self.home['name']} of type {self.home['type']}: {self.prob.status}")
            self.temp_in_min -= 0.2
            self.temp_wh_min -= 0.2
            self.solve_type_problem()
            n_iterations +=1

        if not self.solved:
            self.error_handler()
            # try:
            #     self.get_min_hvac_setbacks()
            #     self.get_min_wh_setbacks()
            #     self.get_alt_p_grid()
            # except:
            #     pass

        end_slice = max(1, self.sub_subhourly_steps)
        # self.log.logger.info(f"Status for {self.home['name']}: {self.prob.status}")
        self.optimal_vals = {
            "p_grid_opt": np.average(self.p_grid.value[0:end_slice]),
            "p_load_opt": np.average(self.p_load.value[0:end_slice]),
            "temp_in_opt": self.temp_in.value[end_slice],
            "temp_wh_opt": self.temp_wh.value[end_slice],
            "hvac_cool_on_opt": np.average(self.hvac_cool_on.value[0:end_slice]),
            "hvac_heat_on_opt": np.average(self.hvac_heat_on.value[0:end_slice]),
            "wh_heat_on_opt": np.average(self.wh_heat_on.value[0:end_slice]),
            "cost_opt": np.average(self.cost.value[0:end_slice])
        }
        if 'pv' in self.type:
            # self.log.logger.debug("Adding pv optimal vals.")
            self.optimal_vals["p_pv_opt"] = np.average(self.p_pv.value[0:end_slice])
            self.optimal_vals["u_pv_curt_opt"] = np.average(self.u_pv_curt.value[0:end_slice])
        if 'battery' in self.type:
            # self.log.logger.debug("Adding battery optimal vals.")
            self.optimal_vals["e_batt_opt"] = self.e_batt.value[end_slice]
            self.optimal_vals["p_batt_ch"] = np.average(self.p_batt_ch.value[0:end_slice])
            self.optimal_vals["p_batt_disch"] = np.average(self.p_batt_disch.value[0:end_slice])
        # self.log.logger.info(f"MPC solved with status {self.prob.status} for {self.home['name']}; {self.optimal_vals}")

    def mpc_base(self):
        """
        Type specific routine for setting up a CVXPY optimization problem.
        self.home == "base"
        :return:
        """
        self.add_base_constraints()
        self.set_base_p_grid()
        self.solve_mpc()

    def mpc_battery(self):
        """
        Type specific routine for setting up a CVXPY optimization problem.
        self.home == "battery_only"
        :return:
        """
        self.setup_battery_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.set_battery_only_p_grid()
        self.solve_mpc()

    def mpc_pv(self):
        """
        Type specific routine for setting up a CVXPY optimization problem.
        self.home == "pv_only"
        :return:
        """
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_pv_constraints()
        self.set_pv_only_p_grid()
        self.solve_mpc()

    def mpc_pv_battery(self):
        """
        Type specific routine for setting up a CVXPY optimization problem.
        self.home == "pv_battery"
        :return:
        """
        self.setup_battery_problem()
        self.setup_pv_problem()
        self.add_base_constraints()
        self.add_battery_constraints()
        self.add_pv_constraints()
        self.set_pv_battery_p_grid()
        self.solve_mpc()

    def redis_get_initial_values(self):
        """
        Collects the values from the outside environment including GHI, OAT, and
        the base price set by the utility.
        :return: None
        """
        self.start_hour_index = self.redis_client.conn.get('start_hour_index')
        self.current_values = self.redis_client.conn.hgetall("current_values")
        self.all_ghi = self.redis_client.conn.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.conn.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.conn.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.conn.lrange('tou', 0, -1)

    def cast_redis_init_vals(self):
        """
        Casts the collected redis values as correct data type.
        :return: None
        """
        self.start_hour_index = int(float(self.start_hour_index))
        self.all_ghi = [float(i) for i in self.all_ghi]
        self.all_oat = [float(i) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]

    def cast_redis_timestep(self):
        """
        Sets the timestep of the current time with respect to total simulation time.
        :return: None
        """
        self.timestep = int(self.current_values["timestep"])

    def cast_redis_curr_rps(self):
        """
        Casts the reward price signal values for the current timestep.
        :return: None
        """
        # print("cast_redis_curr_rps")
        rp = self.redis_client.conn.lrange('reward_price', 0, -1)

        num_agg_steps_seen = int(np.ceil(self.horizon / self.sub_subhourly_steps))
        self.reward_price[:min(len(rp), num_agg_steps_seen)] = rp[:min(len(rp), num_agg_steps_seen)]
        self.reward_price = np.repeat(self.reward_price, self.sub_subhourly_steps)[:self.horizon]
        # print(self.reward_price)
        try:
            self.iteration = int(self.current_values["iteration"])
        except:
            if self.timestep == 0:
                # self.log.logger.debug("Running a non-iterative aggregator agent. Convergence to maximum allowable load not guarantueed.")
                print("non iterative agent")
            else:
                pass
        # self.log.logger.info(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def set_vals_for_current_run(self):
        """
        Slices cast values of the environmental values for the current timestep.
        :return: None
        """
        # print("set_vals_for_current_run")
        start_slice = self.start_hour_index + self.timestep
        # Need to extend 1 timestep past horizon for OAT slice
        end_slice = start_slice + (self.horizon // self.sub_subhourly_steps) + 1
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.oat_current = self.all_oat[start_slice:end_slice]
        self.tou_current = self.all_tou[start_slice:end_slice]

    def solve_type_problem(self):
        """
        Selects routine for MPC optimization problem setup and solve using home type.
        :return: None
        """
        # print("solving type problem")
        self.type = self.home['type']
        if self.type == "base":
            self.mpc_base()
        elif self.type == "battery_only":
            self.mpc_battery()
        elif self.type == "pv_only":
            self.mpc_pv()
        elif self.type == "pv_battery":
            self.mpc_pv_battery()

    def run_home(self, home):
        """
        Intended for parallelization in parent class (e.g. aggregator); runs a
        single MPCCalc home.
        :return: None
        """
        # print("setting home")
        self.home = home
        # self.log = self.home['log']
        # print("getting init vals")
        self.redis_get_initial_values()
        # print("casting init vals")
        self.cast_redis_init_vals()
        # print("casting timestep")
        self.cast_redis_timestep()

        if self.timestep > 0:
            self.redis_get_prev_optimal_vals()

        self.setup_base_problem()
        self.solve_type_problem()
        self.cleanup_and_finish()
        self.redis_write_optimal_vals()
        # print("solved")

    def forecast_home(self, home):
        """
        Intended for parallelization in parent class (e.g. aggregator); runs a
        single MPCCalc home.
        For use of aggregator agent only; does not implement the solution on a house-by-house
        basis, only tells the forecasting agent what the expected p_grid_agg is for the community.
        :return: list (type float) of length self.horizon
        """
        self.home = home
        # self.log = self.home['log']
        self.redis_get_initial_values()
        self.cast_redis_init_vals()
        self.cast_redis_timestep()

        if self.timestep > 0:
            self.redis_get_prev_optimal_vals()

        self.setup_base_problem()
        self.solve_type_problem()
        self.cleanup_and_finish()

        return self.p_grid.value.tolist()
