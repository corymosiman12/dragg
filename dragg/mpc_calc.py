import os
import numpy as np
import cvxpy as cp
# from redis import StrictRedis
import redis
# import scipy.stats
import logging
import pathos
from collections import defaultdict
import json
from copy import deepcopy

from dragg.redis_client import RedisClient
from dragg.logger import Logger

def manage_home(home):
    """
    Calls class method as a top level function (picklizable by pathos)
    :return: None
    """
    home.run_home()
    return

class MPCCalc:
    def __init__(self, home):
        """
        params
        home: Dictionary with keys for HVAC, WH, and optionally PV, battery parameters
        """
        self.home = home  # reset every time home retrieved from Queue
        self.name = home['name']
        self.type = self.home['type']  # reset every time home retrieved from Queue
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
        self.temp_in_ev = None
        self.temp_wh_ev = None
        self.p_grid = None
        self.hvac_cool_on = None
        self.hvac_heat_on = None
        self.wh_heat_on = None
        self.spp = None
        self.oat_forecast = None
        self.ghi_forecast = None
        self.temp_wh_min = None
        self.temp_wh_max = None
        self.temp_in_min = None
        self.temp_in_max = None
        self.optimal_vals = {}
        self.stored_optimal_vals = {
           "p_grid_opt": None,
           "forecast_p_grid_opt": None,
           "p_load_opt": None,
           "temp_in_opt": None,
           "temp_wh_ev_opt": None,
           "hvac_cool_on_opt": None,
           "hvac_heat_on_opt": None,
           "wh_heat_on_opt": None,
           "cost_opt": None,
           "t_in_min": None,
           "t_in_max": None
        }
        self.counter = 0
        self.iteration = None
        self.assumed_wh_draw = None
        self.prev_optimal_vals = None  # set after timestep > 0, set_vals_for_current_run
        self.timestep = 0
        self.p_grid_opt = None

        # setup cvxpy verbose solver
        self.verbose_flag = os.environ.get('VERBOSE','False')
        if not self.verbose_flag.lower() == 'true':
            self.verbose_flag = False
        else:
            self.verbose_flag = True

        # calls redis and gets all the environmental variables from beginning of simulation to end
        self.initialize_environmental_variables()

        # setup the base variables of HVAC and water heater
        self.setup_base_problem()
        if 'battery' in self.type:
            # setup the battery variables
            self.setup_battery_problem()
        if 'pv' in self.type:
            # setup the pv variables
            self.setup_pv_problem()

    def redis_write_optimal_vals(self):
        """
        Sends the optimal values for each home to the redis server.
        :return: None
        """
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        self.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        key = self.name
        for field, value in self.optimal_vals.items():
            # self.log.info(f"field {value}")
            self.redis_client.conn.hset(key, field, value)

        self.log.removeHandler(fh)

    def redis_get_prev_optimal_vals(self):
        """
        Collects starting point environmental values for all homes (such as current temperature).
        :return: None
        """
        key = self.name
        self.prev_optimal_vals = self.redis_client.conn.hgetall(key)

    def initialize_environmental_variables(self):
        self.redis_client = RedisClient()

        # collect all values necessary
        self.start_hour_index = self.redis_client.conn.get('start_hour_index')
        self.all_ghi = self.redis_client.conn.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.conn.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.conn.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.conn.lrange('tou', 0, -1)
        self.base_cents = float(self.all_tou[0])

        # cast all values to proper type
        self.start_hour_index = int(float(self.start_hour_index))
        self.all_ghi = [float(i) for i in self.all_ghi]
        self.all_oat = [float(i) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]

    def setup_base_problem(self):
        """
        Sets variable objects for CVX optimization problem. Includes "base home"
        systems of HVAC and water heater.
        :return: None
        """
        # Set up the solver parameters
        solvers = {"GUROBI": cp.GUROBI, "GLPK_MI": cp.GLPK_MI, "ECOS": cp.ECOS}
        try:
            self.solver = solvers[self.home['hems']['solver']]
        except:
            self.solver = cp.GLPK_MI

        # Set up the horizon for the MPC calc (min horizon = 1, no MPC)
        self.sub_subhourly_steps = max(1, int(self.home['hems']['sub_subhourly_steps']))
        self.dt = max(1, int(self.home['hems']['hourly_agg_steps']))
        self.horizon = max(1, int(self.home['hems']['horizon'] * self.dt))
        self.h_plus = self.horizon + 1
        self.discount = float(self.home['hems']['discount_factor'])
        
        # Set up the HEMS scheule
        self.occ_on = [False] * self.sub_subhourly_steps * self.dt * 24
        occ_sched_pairs = [self.home['hems']['weekday_occ_schedule'][2*i:2*i+2] for i in range(len(self.home['hems']['weekday_occ_schedule'])//2)]
        for occ_period in occ_sched_pairs: # occ_schedule must be a list of lists
            occ_period = [int(x) for x in occ_period]
            int_occ_schedule = [int(x) for x in np.multiply(self.sub_subhourly_steps * self.dt, occ_period)]
            if occ_period[1] < occ_period[0]: # if period of occupancy is overnight
                int_occ_schedule = [int_occ_schedule[0], None, 0, int_occ_schedule[1]]
            for i in range(len(int_occ_schedule)//2):
                start = int_occ_schedule[2*i]
                end = int_occ_schedule[2*i+1]
                len_period = end - start if end else (24 * self.sub_subhourly_steps * self.dt)-start
                self.occ_on[start:end] = [True]*len_period

        # Initialize RP structure so that non-forecasted RPs have an expected value of 0.
        self.reward_price = np.zeros(self.horizon)

        self.home_r = cp.Constant(float(self.home["hvac"]["r"]))
        self.home_c = cp.Constant(float(self.home["hvac"]["c"]) * 1000)
        self.hvac_p_c = cp.Constant(float(self.home["hvac"]["p_c"]) / self.sub_subhourly_steps)
        self.hvac_p_h = cp.Constant((float(self.home["hvac"]["p_h"])) / self.sub_subhourly_steps)
        self.wh_r = cp.Constant(float(self.home["wh"]["r"]) * 1000)
        self.wh_p = cp.Constant(float(self.home["wh"]["p"]) / self.sub_subhourly_steps)

        # Define optimization variables
        self.p_load = cp.Variable(self.horizon)
        self.temp_in_ev = cp.Variable(self.h_plus)
        self.temp_in = cp.Variable(1)
        self.temp_wh_ev = cp.Variable(self.h_plus)
        self.temp_wh = cp.Variable(1)
        self.p_grid = cp.Variable(self.horizon)
        self.hvac_cool_on = cp.Variable(self.horizon, integer=True)
        self.hvac_heat_on = cp.Variable(self.horizon, integer=True)
        self.wh_heat_on = cp.Variable(self.horizon, integer=True)

        # Water heater temperature constraints
        self.temp_wh_min = cp.Constant(float(self.home["wh"]["temp_wh_min"]))
        self.temp_wh_max = cp.Constant(float(self.home["wh"]["temp_wh_max"]))
        self.temp_wh_sp = cp.Constant(float(self.home["wh"]["temp_wh_sp"]))
        self.t_wh_init = float(self.home["wh"]["temp_wh_init"])
        self.wh_size = float(self.home["wh"]["tank_size"])
        self.tap_temp = 15 # assumed cold tap water is about 55 deg F

        wh_capacitance = self.wh_size * 4.2 # kJ/deg C
        self.wh_c = cp.Constant(wh_capacitance)

        # Home temperature constraints
        # create temp bounds based on occ schedule, array should be one week long
        occ_t_in_min = float(self.home["hvac"]["temp_in_min"])
        unocc_t_in_min = float(self.home["hvac"]["temp_in_min"]) - 2#float(self.home["hvac"]["temp_setback_delta"])
        occ_t_in_max = float(self.home["hvac"]["temp_in_max"])
        unocc_t_in_max = float(self.home["hvac"]["temp_in_max"]) + 2#float(self.home["hvac"]["temp_setback_delta"])
        self.t_deadband = occ_t_in_max - occ_t_in_min
        self.t_in_min = [occ_t_in_min if i else unocc_t_in_min for i in self.occ_on] * 2
        self.t_in_max = [occ_t_in_max if i else unocc_t_in_max for i in self.occ_on] * 2
        self.t_in_init = float(self.home["hvac"]["temp_in_init"])

        self.max_load = (max(self.hvac_p_c.value, self.hvac_p_h.value) + self.wh_p.value) * self.sub_subhourly_steps

    def water_draws(self):
        draw_sizes = (self.horizon // self.dt + 1) * [0] + self.home["wh"]["draw_sizes"]
        raw_draw_size_list = draw_sizes[(self.timestep // self.dt):(self.timestep // self.dt) + (self.horizon // self.dt + 1)]
        raw_draw_size_list = (np.repeat(raw_draw_size_list, self.dt) / self.dt).tolist()
        draw_size_list = raw_draw_size_list[:self.dt]
        for i in range(self.dt, self.h_plus):
            draw_size_list.append(np.average(raw_draw_size_list[i-1:i+2]))

        self.draw_size = draw_size_list
        df = np.divide(self.draw_size, self.wh_size)
        self.draw_frac = cp.Constant(df)
        self.remainder_frac = cp.Constant(1-df)

    def set_environmental_variables(self):
        """
        Slices cast values of the environmental values for the current timestep.
        :return: None
        """
        start_slice = self.start_hour_index + self.timestep
        end_slice = start_slice + self.horizon + 1 # Need to extend 1 timestep past horizon for OAT slice

        # Get the current values from a list of all values
        self.ghi_current = self.all_ghi[start_slice:end_slice]
        self.ghi_current_ev = deepcopy(self.ghi_current)
        ghi_noise = np.multiply(self.ghi_current[1:], 0.01 * np.power(1.3*np.ones(self.horizon), np.arange(self.horizon)))
        self.ghi_current_ev[1:] = np.add(self.ghi_current[1:], ghi_noise)

        self.oat_current = self.all_oat[start_slice:end_slice]
        self.oat_current_ev = deepcopy(self.oat_current)
        oat_noise = np.multiply(np.power(1.1*np.ones(self.horizon), np.arange(self.horizon)), np.random.randn(self.horizon))
        self.oat_current_ev[1:] = np.add(self.oat_current[1:], oat_noise)

        self.tou_current = self.all_tou[start_slice:end_slice]
        self.base_price = np.array(self.tou_current, dtype=float)

        # Set values as cvxpy values
        self.oat_forecast = cp.Constant(self.oat_current)
        self.ghi_forecast = cp.Constant(self.ghi_current)
        self.cast_redis_curr_rps()

    def setup_battery_problem(self):
        """
        Adds CVX variables for battery subsystem in battery and battery_pv homes.
        :return: None
        """
        # Define constants
        self.batt_max_rate = cp.Constant(float(self.home["battery"]["max_rate"]))
        self.batt_cap_total = cp.Constant(float(self.home["battery"]["capacity"]))
        self.batt_cap_min = cp.Constant(float(self.home["battery"]["capacity_lower"]) * self.batt_cap_total.value)
        self.batt_cap_max = cp.Constant(float(self.home["battery"]["capacity_upper"]) * self.batt_cap_total.value)
        self.batt_ch_eff = cp.Constant(float(self.home["battery"]["ch_eff"]))
        self.batt_disch_eff = cp.Constant(float(self.home["battery"]["disch_eff"]))

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

    def get_initial_conditions(self):
        self.water_draws()

        if self.timestep == 0:
            self.initialize_environmental_variables()

            self.temp_in_init = cp.Constant(self.t_in_init)
            self.temp_wh_init = cp.Constant((self.t_wh_init*(self.wh_size - self.draw_size[0]) + self.tap_temp * self.draw_size[0]) / self.wh_size)

            if 'battery' in self.type:
                self.e_batt_init = cp.Constant(float(self.home["battery"]["e_batt_init"]) * self.batt_cap_total.value)
                self.p_batt_ch_init = cp.Constant(0)

            self.counter = 0

        else:
            self.temp_in_init = cp.Constant(float(self.prev_optimal_vals["temp_in_opt"]))
            self.temp_wh_init = cp.Constant((float(self.prev_optimal_vals["temp_wh_opt"])*(self.wh_size - self.draw_size[0]) + self.tap_temp * self.draw_size[0]) / self.wh_size)

            if 'battery' in self.type:
                self.e_batt_init = cp.Constant(float(self.home["battery"]["e_batt_init"]))
                self.e_batt_init = cp.Constant(float(self.prev_optimal_vals["e_batt_opt"]))
                self.p_batt_ch_init = cp.Constant(float(self.prev_optimal_vals["p_batt_ch"])
                                                - float(self.prev_optimal_vals["p_batt_disch"]))

            self.counter = int(self.prev_optimal_vals["solve_counter"])

    def add_base_constraints(self):
        """
        Creates the system dynamics for thermal energy storage systems: HVAC and
        water heater.
        :return: None
        """
        self.hvac_heat_min = 0
        self.hvac_cool_min = 0
        self.wh_heat_max = self.sub_subhourly_steps
        self.wh_heat_min = 0
        # Set constraints on HVAC by season
        if max(self.oat_current_ev) <= 30: # "winter"
            self.hvac_heat_max = self.sub_subhourly_steps
            self.hvac_cool_max = 0

        else: # "summer"
            self.hvac_heat_max = 0
            self.hvac_cool_max = self.sub_subhourly_steps
        
        start = (self.timestep * self.sub_subhourly_steps * self.dt) % (24 * self.sub_subhourly_steps * self.dt)
        stop = start + self.horizon#(self.timestep * self.sub_subhourly_steps * self.dt ) % 24 + self.horizon
        self.t_in_max_current = cp.Constant(self.t_in_max[start:stop])
        self.t_in_min_current = cp.Constant(self.t_in_min[start:stop])
        
        self.constraints = [
            # Indoor air temperature constraints
            # self.temp_in_ev = indoor air temperature expected value (prediction)
            self.temp_in_ev[0] == self.temp_in_init,
            self.temp_in_ev[1:self.h_plus] == self.temp_in_ev[0:self.horizon]
                                            + 3600 * ((((self.oat_forecast[1:self.h_plus] - self.temp_in_ev[0:self.horizon]) / self.home_r))
                                            - self.hvac_cool_on * self.hvac_p_c
                                            + self.hvac_heat_on * self.hvac_p_h) / (self.home_c * self.dt),
            self.temp_in_ev[1:self.h_plus] >= self.t_in_min_current,
            self.temp_in_ev[1:self.h_plus] <= self.t_in_max_current,

            self.temp_in == self.temp_in_init
                            + 3600 * (((self.oat_current[1] - self.temp_in_init) / self.home_r)
                            - self.hvac_cool_on[0] * self.hvac_p_c
                            + self.hvac_heat_on[0] * self.hvac_p_h) / (self.home_c * self.dt),
            self.temp_in <= self.t_in_max_current[0],
            self.temp_in >= self.t_in_min_current[0],

            # Hot water heater contraints, expected value after approx waterdraws
            self.temp_wh_ev[0] == self.temp_wh_init,
            self.temp_wh_ev[1:] == (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:self.horizon]) + self.draw_frac[1:]*self.tap_temp)
                                + 3600 * ((((self.temp_in_ev[1:] - (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:self.horizon]) + self.draw_frac[1:]*self.tap_temp)) / self.wh_r))
                                + self.wh_heat_on * self.wh_p) / (self.wh_c * self.dt),
            self.temp_wh_ev >= self.temp_wh_min,
            self.temp_wh_ev <= self.temp_wh_max,

            self.temp_wh == self.temp_wh_init
                            + 3600 * (((self.temp_in_ev[1] - self.temp_wh_init) / self.wh_r)
                            + self.wh_heat_on[0] * self.wh_p) / (self.wh_c * self.dt),
            self.temp_wh >= self.temp_wh_min,
            self.temp_wh <= self.temp_wh_max,

            self.p_load ==  self.sub_subhourly_steps * (self.hvac_p_c * self.hvac_cool_on + self.hvac_p_h * self.hvac_heat_on + self.wh_p * self.wh_heat_on),

            self.hvac_cool_on <= self.hvac_cool_max,
            self.hvac_cool_on >= self.hvac_cool_min,
            self.hvac_heat_on <= self.hvac_heat_max,
            self.hvac_heat_on >= self.hvac_heat_min,
            self.wh_heat_on <= self.wh_heat_max,
            self.wh_heat_on >= self.wh_heat_min
        ]

        # set total price for electricity
        self.total_price = cp.Constant(np.array(self.reward_price, dtype=float) + self.base_price[:self.horizon])

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
        ]

    def add_pv_constraints(self):
        """
        Creates the system dynamics for photovoltaic generation. (Using measured GHI.)
        :return: None
        """
        self.constraints += [
            # PV constraints.  GHI provided in W/m2 - convert to kWh
            self.p_pv == self.pv_area * self.pv_eff * cp.multiply(self.ghi_forecast[0:self.horizon], (1 - self.u_pv_curt)) / 1000,
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
            self.p_grid == self.p_load # where p_load = p_hvac + p_wh
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
            self.p_grid == self.p_load + self.sub_subhourly_steps * (self.p_batt_ch + self.p_batt_disch)
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
            self.p_grid == self.p_load - self.sub_subhourly_steps * self.p_pv
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
            self.p_grid == self.p_load + self.sub_subhourly_steps * (self.p_batt_ch + self.p_batt_disch - self.p_pv)
        ]

    def solve_mpc(self):
        """
        Sets the objective function of the Home Energy Management System to be the
        minimization of cost over the MPC time horizon and solves via CVXPY.
        Used for all home types.
        :return: None
        """
        self.cost = cp.Variable(self.horizon)
        self.wh_weighting = 10
        self.objective = cp.Variable(self.horizon)
        self.constraints += [self.cost == cp.multiply(self.total_price, self.p_grid)] # think this should work
        self.weights = cp.Constant(np.power(self.discount*np.ones(self.horizon), np.arange(self.horizon)))
        self.obj = cp.Minimize(cp.sum(cp.multiply(self.cost, self.weights))) #+ self.wh_weighting * cp.sum(cp.abs(self.temp_wh_max - self.temp_wh_ev))) #cp.sum(self.temp_wh_sp - self.temp_wh_ev))
        self.prob = cp.Problem(self.obj, self.constraints)
        if not self.prob.is_dcp():
            # self.log.error("Problem is not DCP")
            print("problem is not dcp")
        # try:
        #     print('this')
        self.prob.solve(solver=self.solver, verbose=self.verbose_flag)
        # self.solved = True
        # except:
        #     print('that')
        #     self.solved = False
        return

    def implement_presolve(self):
        constraints = [
            # Indoor air temperature constraints
            self.temp_in_ev[0] == self.temp_in_init,
            self.temp_in_ev[1:self.h_plus] == self.temp_in_ev[0:self.horizon]
                                            + (((self.oat_forecast[1:self.h_plus] - self.temp_in_ev[0:self.horizon]) / self.home_r)
                                            - self.hvac_cool_on * self.hvac_p_c
                                            + self.hvac_heat_on * self.hvac_p_h) / (self.home_c * self.dt),

            # Hot water heater contraints
            self.temp_wh_ev[0] == self.temp_wh_init,
            self.temp_wh_ev[1:] == self.temp_wh_ev[:self.horizon]
                                + (((self.temp_in_ev[1:self.h_plus] - self.temp_wh_ev[:self.horizon]) / self.wh_r)
                                + self.wh_heat_on * self.wh_p) / (self.wh_c * self.dt),

            self.hvac_cool_on == np.array(self.presolve_hvac_cool_on, dtype=np.double),
            self.hvac_heat_on == np.array(self.presolve_hvac_heat_on, dtype=np.double),
            self.wh_heat_on == np.array(self.presolve_wh_heat_on, dtype=np.double)
        ]

    def cleanup_and_finish(self):
        """
        Resolves .solve_mpc() with error handling and collects all data on solver.
        :return: None
        """
        end_slice = max(1, self.sub_subhourly_steps)
        opt_keys = {"p_grid_opt", "forecast_p_grid_opt", "p_load_opt", "temp_in_ev_opt", "temp_wh_ev_opt", "hvac_cool_on_opt", "hvac_heat_on_opt", "wh_heat_on_opt", "cost_opt", "waterdraws", "t_in_min", "t_in_max"}

        i = 0
        while i < 1:
            if self.prob.status == 'optimal': # if the problem has been solved
                self.counter = 0
                self.timestep += 1
                self.stored_optimal_vals = defaultdict()
                self.stored_optimal_vals["p_grid_opt"] = (self.p_grid.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["forecast_p_grid_opt"] = (self.p_grid.value[1:] / self.sub_subhourly_steps).tolist() + [0]
                self.stored_optimal_vals["p_load_opt"] = (self.p_load.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["temp_in_ev_opt"] = (self.temp_in_ev.value[1:]).tolist()
                self.stored_optimal_vals["temp_in_opt"] = self.temp_in.value.tolist()
                self.stored_optimal_vals["temp_wh_ev_opt"] = (self.temp_wh_ev.value[1:]).tolist()
                self.stored_optimal_vals["temp_wh_opt"] = self.temp_wh.value.tolist()
                self.stored_optimal_vals["hvac_cool_on_opt"] = (self.hvac_cool_on.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["hvac_heat_on_opt"] = (self.hvac_heat_on.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["wh_heat_on_opt"] = (self.wh_heat_on.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["cost_opt"] = (self.cost.value).tolist()
                self.stored_optimal_vals["waterdraws"] = self.draw_size
                self.stored_optimal_vals["t_in_min"] = self.t_in_min_current.value.tolist()
                self.stored_optimal_vals["t_in_max"] = self.t_in_max_current.value.tolist()
                self.all_optimal_vals = {}

                if 'pv' in self.type:
                    self.stored_optimal_vals['p_pv_opt'] = (self.p_pv.value).tolist()
                    self.stored_optimal_vals['u_pv_curt_opt'] = (self.u_pv_curt.value).tolist()
                    opt_keys.update(['p_pv_opt', 'u_pv_curt_opt'])
                if 'battery' in self.type:
                    self.stored_optimal_vals['e_batt_opt'] = (self.e_batt.value).tolist()[1:]
                    self.stored_optimal_vals['p_batt_ch'] = (self.p_batt_ch.value).tolist()
                    self.stored_optimal_vals['p_batt_disch'] = (self.p_batt_disch.value).tolist()
                    opt_keys.update(['p_batt_ch', 'p_batt_disch', 'e_batt_opt'])

                for k in opt_keys:
                    if not k == "waterdraws":
                        self.optimal_vals[k] = self.stored_optimal_vals[k][0]
                    else:
                        self.optimal_vals[k] = self.stored_optimal_vals[k][0]
                    for j in range(self.horizon):
                        self.optimal_vals[f"{k}_{j}"] = self.stored_optimal_vals[k][j]
                self.optimal_vals["temp_wh_opt"] = self.stored_optimal_vals["temp_wh_opt"][0]
                self.optimal_vals["temp_in_opt"] = self.stored_optimal_vals["temp_in_opt"][0]
                self.optimal_vals["t_in_min"] = self.stored_optimal_vals["t_in_min"][0]
                self.optimal_vals["t_in_max"] = self.stored_optimal_vals["t_in_max"][0]
                self.optimal_vals["correct_solve"] = 1
                self.optimal_vals["solve_counter"] = 0
                # self.log.debug(f"MPC solved with status {self.prob.status} for {self.name}")
                return
            else:
                # self.implement_presolve()
                self.counter += 1
                # self.log.warning(f"Unable to solve for house {self.name}. Reverting to optimal solution from last feasible timestep, t-{self.counter}.")
                self.optimal_vals["correct_solve"] = 0

                if self.counter < self.horizon and self.timestep > 0:
                    for k in opt_keys:
                        self.optimal_vals[k] = self.prev_optimal_vals[f"{k}_{self.counter}"]

                    self.presolve_wh_heat_on = float(self.optimal_vals["wh_heat_on_opt"][0])
                    self.presolve_hvac_cool_on = float(self.optimal_vals["hvac_cool_on_opt"][0])
                    self.presolve_hvac_heat_on = float(self.optimal_vals["hvac_heat_on_opt"][0])

                    new_temp_in = float(self.temp_in_init.value
                                        + 3600 * ((((self.oat_current[1] - self.temp_in_init.value) / self.home_r.value))
                                        - self.presolve_hvac_cool_on * self.hvac_p_c.value
                                        + self.presolve_hvac_heat_on * self.hvac_p_h.value) / (self.home_c.value * self.dt))
                    new_temp_wh = float(self.temp_wh_init.value
                                        + 3600 * ((((new_temp_in - self.temp_wh_init.value) / self.wh_r.value))
                                        + self.presolve_wh_heat_on * self.wh_p.value) / (self.wh_c.value * self.dt))

                    if new_temp_in > self.t_in_max_current[0].value:
                        self.presolve_hvac_heat_on = self.hvac_heat_min
                        self.presolve_hvac_cool_on = self.hvac_cool_max
                    elif new_temp_in < self.t_in_min_current[0].value:
                        self.presolve_hvac_heat_on = self.hvac_heat_max
                        self.presolve_hvac_cool_on = self.hvac_cool_min

                    if new_temp_wh < self.temp_wh_min.value:
                        self.presolve_wh_heat_on = self.wh_heat_max

                else:
                    self.counter = int(np.clip(self.counter, self.horizon, None))
                    if self.temp_in_init.value > self.temp_in_max.value:
                        self.presolve_hvac_heat_on = self.hvac_heat_min
                        self.presolve_hvac_cool_on = self.hvac_cool_max
                    elif self.temp_in_init.value < self.temp_in_min.value:
                        self.presolve_hvac_heat_on = self.hvac_heat_max
                        self.presolve_hvac_cool_on = self.hvac_cool_min
                    else:
                        self.presolve_hvac_heat_on = self.hvac_heat_min
                        self.presolve_hvac_cool_on = self.hvac_cool_min

                    if self.temp_wh_init.value < self.temp_wh_min.value:
                        self.presolve_wh_heat_on = self.wh_heat_max
                    else:
                        self.presolve_wh_heat_on = self.wh_heat_min

                new_temp_in = float(self.temp_in_init.value
                                    + 3600 * ((((self.oat_current[1] - self.temp_in_init.value) / self.home_r.value))
                                    - self.presolve_hvac_cool_on * self.hvac_p_c.value
                                    + self.presolve_hvac_heat_on * self.hvac_p_h.value) / (self.home_c.value * self.dt))
                new_temp_wh = float(self.temp_wh_init.value
                                    + 3600 * (((new_temp_in - self.temp_wh_init.value) / self.wh_r.value)
                                    + (self.presolve_wh_heat_on * self.wh_p.value)) / (self.wh_c.value * self.dt))

                self.optimal_vals["wh_heat_on_opt"] = self.presolve_wh_heat_on / self.sub_subhourly_steps
                self.optimal_vals["hvac_heat_on_opt"] = self.presolve_hvac_heat_on / self.sub_subhourly_steps
                self.optimal_vals["hvac_cool_on_opt"] = self.presolve_hvac_cool_on / self.sub_subhourly_steps
                self.optimal_vals["temp_in_opt"] = new_temp_in
                self.optimal_vals["temp_wh_opt"] = new_temp_wh
                self.optimal_vals["solve_counter"] = self.counter
                self.optimal_vals["p_load_opt"] = self.presolve_wh_heat_on * self.wh_p.value + self.presolve_hvac_cool_on * self.hvac_p_c.value + self.presolve_hvac_heat_on * self.hvac_p_h.value
                self.optimal_vals["forecast_p_grid_opt"] = self.optimal_vals["p_load_opt"]
                self.optimal_vals["waterdraws"] = self.draw_size[0]
                self.optimal_vals["p_grid_opt"] = self.optimal_vals["p_load_opt"]
                self.optimal_vals["cost_opt"] = self.optimal_vals["p_grid_opt"] * self.total_price.value[0]
                self.optimal_vals["t_in_min"] = self.t_in_min_current.value[0]
                self.optimal_vals["t_in_max"] = self.t_in_max_current.value[0]
                i+=1
                pass

    def add_type_constraints(self):
        self.add_base_constraints()
        if 'pv' in self.type:
            self.add_pv_constraints()
        if 'batt' in self.type:
            self.add_battery_constraints()

    def set_type_p_grid(self):
        if self.type == "base":
            self.set_base_p_grid()
        elif self.type == "pv_only":
            self.set_pv_only_p_grid()
        elif self.type == "battery_only":
            self.set_battery_only_p_grid()
        else:
            self.set_pv_battery_p_grid()

    def redis_get_initial_values(self):
        """
        Collects the values from the outside environment including GHI, OAT, and
        the base price set by the utility.
        :return: None
        """
        self.current_values = self.redis_client.conn.hgetall("current_values")

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
        rp = self.redis_client.conn.lrange('reward_price', 0, -1)
        self.reward_price = rp[:self.horizon]
        # self.log.debug(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def solve_type_problem(self):
        """
        Selects routine for MPC optimization problem setup and solve using home type.
        :return: None
        """
        self.set_environmental_variables()
        self.add_type_constraints()
        self.set_type_p_grid()
        self.solve_mpc()
        return

    def run_home(self):
        """
        Intended for parallelization in parent class (e.g. aggregator); runs a
        single MPCCalc home.
        :return: None
        """
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        fh.setLevel(logging.WARN)

        # self.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        self.redis_client = RedisClient()
        self.redis_get_initial_values()
        self.cast_redis_timestep()

        if self.timestep > 0:
            self.redis_get_prev_optimal_vals()

        self.get_initial_conditions()
        self.solve_type_problem()
        self.cleanup_and_finish()
        self.redis_write_optimal_vals()

        # self.log.removeHandler(fh)
