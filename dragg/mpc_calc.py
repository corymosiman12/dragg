import os
import numpy as np
import cvxpy as cp
import redis
import logging
import pathos
from collections import defaultdict
import json
from copy import deepcopy

import dragg.redis_client as rc # from dragg.redis_client import connection#RedisClient
from dragg.logger import Logger
from dragg.devices import *

REDIS_URL = "redis://localhost"

def manage_home(home):
    """
    Calls class method as a top level function (picklizable by pathos)
    :return: None
    """
    home.run_home()
    return

class MPCCalc:
    def __init__(self, home, redis_url=REDIS_URL):
        """
        :paremeter home: dictionary with keys for HVAC, WH, and optionally PV, battery parameters
        :redis_url: optional override of the Redis host URL (must align with MPCCalc REDIS_URL) 
        """
        self.redis_url = redis_url
        self.home = home  # reset every time home retrieved from Queue
        self.name = home['name']
        self.type = self.home['type']  # reset every time home retrieved from Queue
        self.devices = ['hvac', 'wh', 'ev']
        if 'batt' in self.type:
            self.devices += ['battery']
        if 'pv' in self.type:
            self.devices += ['pv']
        self.start_hour_index = None  # set once upon thread init
        self.current_values = None  # set once upon thread init
        self.all_ghi = None  # list, all values in the GHI list, set once upon thread init
        self.all_oat = None  # list, all values in the OAT list, set once upon thread init
        self.all_spp = None  # list, all values in the SPP list, set once upon thread init
        self.p_load = None
        self.plug_load = None
        self.p_grid = None
        self.spp = None
        self.optimal_vals = {
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
        self.p_grid_opt = None
        self.timestep = 0
        self.start_index = None

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

        self.ev_override_profile = None

    def redis_write_optimal_vals(self):
        """
        Sends the optimal values for each home to the redis server.
        :return: None
        """
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        self.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        # optimization variables
        key = self.name
        for field, value in self.optimal_vals.items():
            try:
                self.redis_client.hset(key, field, value)
            except:
                print(key, field, value)

        self.log.removeHandler(fh)

    def redis_get_prev_optimal_vals(self):
        """
        Collects starting point environmental values for all homes (such as current temperature).
        :return: None
        """
        key = self.name
        self.prev_optimal_vals = self.redis_client.hgetall(key)

    def initialize_environmental_variables(self):
        self.redis_client = rc.connection(self.redis_url)

        # collect all values necessary
        self.start_hour_index = self.redis_client.get('start_hour_index')
        self.all_ghi = self.redis_client.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.lrange('tou', 0, -1)
        self.base_cents = float(self.all_tou[0])

        # cast all values to proper type
        self.start_hour_index = int(float(self.start_hour_index))
        self.all_ghi = [float(i) for i in self.all_ghi]
        self.all_oat = [float(i) for i in self.all_oat]
        self.all_spp = [float(i) for i in self.all_spp]

    def _setup_hvac_problem(self):
        self.hvac = HVAC(self)
        self.opt_keys.update(self.hvac.opt_keys)

    def _setup_wh_problem(self):
        self.wh = WH(self)
        self.opt_keys.update(self.wh.opt_keys)

    def setup_base_problem(self):
        """
        Sets variable objects for CVX optimization problem. Includes "base home"
        systems of HVAC and water heater.
        :return: None
        """
        self.opt_keys = set()

        # Set up the solver parameters
        solvers = {"GUROBI": cp.GUROBI, "GLPK_MI": cp.GLPK_MI, "ECOS": cp.ECOS}
        try:
            self.solver = solvers[self.home['hems']['solver']]
        except:
            self.solver = cp.GLPK_MI

        # Set up the horizon for the MPC calc (min horizon = 1, no MPC)
        self.sub_subhourly_steps = max(1, int(self.home['hems']['sub_subhourly_steps']))
        self.dt = max(1, int(self.home['hems']['hourly_agg_steps']))
        self.horizon = max(1, int(self.home['hems']['horizon']) * int(self.dt))
        self.h_plus = self.horizon + 1
        self.discount = float(self.home['hems']['discount_factor'])
        self.dt_frac = 1 / self.dt / self.sub_subhourly_steps
        
        # Initialize RP structure so that non-forecasted RPs have an expected value of 0.
        self.reward_price = np.zeros(self.horizon)

        # Define optimization variables
        self.p_load = cp.Variable(self.horizon)
        self.p_grid = cp.Variable(self.horizon)

        self.r = cp.Constant(float(self.home["hvac"]["r"]))
        self.c = cp.Constant(float(self.home["hvac"]["c"])* 1e7) 
        self.window_eq = cp.Constant(float(self.home["hvac"]["w"]))

        self.occ_on = [False] * self.dt * 24
        occ_sched_pairs = [self.home['hems']['weekday_occ_schedule'][2*i:2*i+2] for i in range(len(self.home['hems']['weekday_occ_schedule'])//2)]
        for occ_period in occ_sched_pairs: # occ_schedule must be a list of lists
            occ_period = [int(x) for x in occ_period]
            int_occ_schedule = [int(x) for x in np.multiply(self.dt, occ_period)]
            if occ_period[1] < occ_period[0]: # if period of occupancy is overnight
                int_occ_schedule = [int_occ_schedule[0], None, 0, int_occ_schedule[1]]
            for i in range(len(int_occ_schedule)//2):
                start = int_occ_schedule[2*i]
                end = int_occ_schedule[2*i+1]
                len_period = end - start if end else (24 * self.dt)-start
                self.occ_on[start:end] = [True]*len_period
        self.leaving_times = [pair[1] * self.dt for pair in occ_sched_pairs]
        self.returning_times = [pair[0] * self.dt for pair in occ_sched_pairs]

        self.max_load = 0
        if 'hvac' in self.devices:
            self._setup_hvac_problem()
            self.max_load += max(self.hvac.p_c.value, self.hvac.p_h.value)
        if 'wh' in self.devices:
            self._setup_wh_problem()
            self.max_load += self.wh.p.value
        if 'ev' in self.devices:
            self._setup_ev_problem()

        self.opt_keys.update({"p_grid_opt", "forecast_p_grid_opt", "p_load_opt", "cost_opt"})
    
    def set_environmental_variables(self):
        """
        Slices cast values of the environmental values for the current timestep.
        
        :return: None
        """
        start_slice = self.start_hour_index + self.timestep
        end_slice = start_slice + self.h_plus # Need to extend 1 timestep past horizon for OAT slice

        # Get the current values from a list of all values
        self.ghi_current = cp.Constant(self.all_ghi[start_slice:end_slice])
        self.ghi_current_ev = deepcopy(self.ghi_current.value)
        ghi_noise = np.multiply(self.ghi_current.value[1:], 0.01 * np.power(1.3*np.ones(self.horizon), np.arange(self.horizon)))
        self.ghi_current_ev[1:] = np.add(self.ghi_current.value[1:], ghi_noise)

        self.oat_current = cp.Constant(self.all_oat[start_slice:end_slice])
        self.oat_current_ev = deepcopy(self.oat_current.value)
        oat_noise = np.multiply(np.power(1.1*np.ones(self.horizon), np.arange(self.horizon)), np.random.randn(self.horizon))
        self.oat_current_ev[1:] = np.add(self.oat_current.value[1:], oat_noise)

        self.tou_current = self.all_tou[start_slice:end_slice]
        self.base_price = np.array(self.tou_current, dtype=float)

        self.cast_redis_curr_rps()

    def _setup_battery_problem(self):
        """
        Adds CVX variables for battery subsystem in battery and battery_pv homes.
        :return: None
        """
        self.battery = Battery(self)
        self.opt_keys.update(self.battery.opt_keys)

    def _setup_ev_problem(self):
        self.ev = EV(self)
        self.leaving_times = [int(i) if not isinstance(i, int) else i for i in self.leaving_times]
        self.returning_times = [int(i) if not isinstance(i, int) else i for i in self.returning_times]
        
        # self.leaving_times = self.ev.leaving_times
        # self.returning_times = self.ev.returning_times
        self.opt_keys.update(self.ev.opt_keys)

    def _setup_pv_problem(self):
        """
        Adds CVX variables for photovoltaic subsystem in pv and battery_pv homes.
        
        :return: None
        """
        self.pv = PV(self)
        self.opt_keys.update(self.pv.opt_keys)

    def get_initial_conditions(self):
        # self.water_draws()

        if self.timestep == 0:
            self.initialize_environmental_variables()

            self.temp_in_init = cp.Constant(float(self.home["hvac"]["temp_in_init"]))
            self.temp_wh_init = cp.Constant((float(self.home["wh"]["temp_wh_init"])*(self.wh.wh_size - self.wh.draw_size[0]) + self.wh.tap_temp * self.wh.draw_size[0]) / self.wh.wh_size)
            self.e_ev_init = cp.Constant(16)
            if 'battery' in self.type:
                self.e_batt_init = cp.Constant(float(self.home["battery"]["e_batt_init"]) * self.batt_cap_total.value)
                self.p_batt_ch_init = cp.Constant(0)

            self.counter = 0

        else:
            self.temp_in_init = cp.Constant(float(self.prev_optimal_vals["temp_in_opt"]))
            self.temp_wh_init = cp.Constant(float(self.prev_optimal_vals["temp_wh_opt"])) #cp.Constant((float(self.prev_optimal_vals["temp_wh_opt"])*(self.wh.wh_size - self.wh.draw_size[0]) + self.wh.tap_temp * self.wh.draw_size[0]) / self.wh.wh_size)
            
            if 'battery' in self.type:
                self.e_batt_init = cp.Constant(float(self.prev_optimal_vals["e_batt_opt"]))

            if 'ev' in self.devices:
                self.e_ev_init = cp.Constant(float(self.prev_optimal_vals["e_ev_opt"]))

            self.counter = int(self.prev_optimal_vals["solve_counter"])

        self.set_environmental_variables()
        self.season = "heating" if max(self.oat_current_ev) <= 27 else "cooling"
        self.add_current_bounds()


    def add_current_bounds(self):
        start = self.timestep % (24 * self.dt)
        stop = start + self.horizon
        self.hvac.t_in_max_current = cp.Constant(self.hvac.t_in_max[start:stop])
        self.hvac.t_in_min_current = cp.Constant(self.hvac.t_in_min[start:stop])

    def add_base_constraints(self):
        """
        Creates the system dynamics for thermal energy storage systems: HVAC and
        water heater.
        
        :return: None
        """
        self.constraints = [
            self.p_load == self.sub_subhourly_steps * (self.hvac.p_elec + self.wh.p * self.wh.heat_on)
        ]
        if 'hvac' in self.devices:
            self.constraints += self.hvac.add_constraints()
        if 'wh' in self.devices:
            self.constraints += self.wh.add_constraints()
        if 'ev' in self.devices:
            self.constraints += self.ev.add_constraints()
        if 'pv' in self.devices:
            # self.add_pv_constraints()
            self.constraints += self.pv.add_constraints()
        if 'battery' in self.devices:
            # self.add_battery_constraints()
            self.constraints += self.battery.add_constraints()
        # set total price for electricity
        self.total_price = cp.Constant(np.array(self.reward_price, dtype=float) + self.base_price[:self.horizon])

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

    def solve_mpc(self, debug=False):
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
        self.prob.solve(solver=self.solver, verbose=self.verbose_flag)
        return

    def cleanup_and_finish(self):
        """
        Resolves .solve_mpc() with error handling and collects all data on solver.
        
        :return: None
        """
        end_slice = max(1, self.sub_subhourly_steps)
        
        i = 0
        while i < 1:
            if self.prob.status == 'optimal': # if the problem has been solved
                self.counter = 0
                # self.timestep += 1
                self.stored_optimal_vals = defaultdict()

                # general base values
                self.stored_optimal_vals["cost_opt"] = (self.cost.value).tolist()
                self.stored_optimal_vals["p_grid_opt"] = (self.p_grid.value / self.sub_subhourly_steps).tolist()
                self.stored_optimal_vals["forecast_p_grid_opt"] = (self.p_grid.value[1:] / self.sub_subhourly_steps).tolist() + [0]
                self.stored_optimal_vals["p_load_opt"] = (self.p_load.value / self.sub_subhourly_steps).tolist()
                
                if 'hvac' in self.devices:
                    self.stored_optimal_vals["temp_in_ev_opt"] = (self.hvac.temp_in_ev.value[1:]).tolist()
                    self.stored_optimal_vals["temp_in_opt"] = self.hvac.temp_in.value.tolist()
                    self.stored_optimal_vals["t_in_min"] = self.hvac.t_in_min_current.value.tolist()
                    self.stored_optimal_vals["t_in_max"] = self.hvac.t_in_max_current.value.tolist()
                    self.stored_optimal_vals["hvac_cool_on_opt"] = (self.hvac.cool_on.value / self.sub_subhourly_steps).tolist()
                    self.stored_optimal_vals["hvac_heat_on_opt"] = (self.hvac.heat_on.value / self.sub_subhourly_steps).tolist()

                if 'wh' in self.devices:
                    self.stored_optimal_vals["temp_wh_ev_opt"] = (self.wh.temp_wh_ev.value[1:]).tolist()
                    self.stored_optimal_vals["temp_wh_opt"] = self.wh.temp_wh.value.tolist()
                    self.stored_optimal_vals["wh_heat_on_opt"] = (self.wh.heat_on.value / self.sub_subhourly_steps).tolist()
                    self.stored_optimal_vals["waterdraws"] = self.wh.draw_size
                
                if 'ev' in self.devices:
                    self.stored_optimal_vals['e_ev_opt'] = (self.ev.e_ev.value).tolist()[1:]
                    self.stored_optimal_vals['p_ev_ch'] = (self.ev.p_ev_ch.value).tolist()
                    self.stored_optimal_vals['p_ev_disch'] = (self.ev.p_ev_disch.value).tolist()
                    self.stored_optimal_vals['p_v2g'] = (self.ev.p_v2g.value).tolist()

                if 'pv' in self.type:
                    self.stored_optimal_vals['p_pv_opt'] = (self.p_pv.value).tolist()
                    self.stored_optimal_vals['u_pv_curt_opt'] = (self.u_pv_curt.value).tolist()

                if 'battery' in self.type:
                    self.stored_optimal_vals['e_batt_opt'] = (self.e_batt.value).tolist()[1:]
                    self.stored_optimal_vals['p_batt_ch'] = (self.p_batt_ch.value).tolist()
                    self.stored_optimal_vals['p_batt_disch'] = (self.p_batt_disch.value).tolist()
                
                for k in self.opt_keys:
                    self.optimal_vals[k] = self.stored_optimal_vals[k][0]
                    for j in range(self.horizon):
                        self.optimal_vals[f"{k}_{j} "] = self.stored_optimal_vals[k][j]

                if 'hvac' in self.devices:
                    self.optimal_vals["temp_in_opt"] = self.stored_optimal_vals["temp_in_opt"][0]

                if 'wh' in self.devices:
                    self.optimal_vals["temp_wh_opt"] = self.stored_optimal_vals["temp_wh_opt"][0]
                
                self.optimal_vals["correct_solve"] = 1
                self.optimal_vals["solve_counter"] = 0
                # self.log.debug(f"MPC solved with status {self.prob.status} for {self.name}")
                return
            
            else:
                self.counter += 1
                # self.log.warning(f"Unable to solve for house {self.name}. Reverting to optimal solution from last feasible timestep, t-{self.counter}.")
                self.optimal_vals["correct_solve"] = 0
                self.optimal_vals["p_load_opt"] = 0 

                if 'hvac' in self.devices:
                    self.hvac.resolve()
                    self.optimal_vals["p_load_opt"] += self.hvac.p_elec.value[0]
                    self.optimal_vals["hvac_heat_on_opt"] = self.hvac.heat_on.value[0] / self.sub_subhourly_steps #self.presolve_hvac_heat_on / self.sub_subhourly_steps
                    self.optimal_vals["hvac_cool_on_opt"] = self.hvac.heat_on.value[0] / self.sub_subhourly_steps #self.presolve_hvac_cool_on / self.sub_subhourly_steps
                    self.optimal_vals["temp_in_opt"] = self.hvac.temp_in.value[0]
                    self.optimal_vals["t_in_min"] = self.hvac.t_in_min_current.value[0]
                    self.optimal_vals["t_in_max"] = self.hvac.t_in_max_current.value[0]

                if 'wh' in self.devices:
                    self.wh.resolve()
                    self.optimal_vals["p_load_opt"] += self.wh.heat_on.value[0] * self.wh.p.value 
                    self.optimal_vals["wh_heat_on_opt"] = self.wh.heat_on.value[0] / self.sub_subhourly_steps #self.presolve_wh_heat_on / self.sub_subhourly_steps
                    self.optimal_vals["temp_wh_opt"] = self.wh.temp_wh.value[0] #new_temp_wh
                    self.optimal_vals["waterdraws"] = self.wh.draw_size[0]

                if 'ev' in self.devices:
                    self.ev.resolve()
                    self.optimal_vals["p_ev_ch"] = self.ev.p_ev_ch.value.tolist()[0]
                    self.optimal_vals["p_ev_disch"] = self.ev.p_ev_disch.value.tolist()[0]
                    self.optimal_vals["p_ev_v2g"] = self.ev.p_v2g.value.tolist()[0]
                    self.optimal_vals["e_ev_opt"] = self.ev.e_ev.value.tolist()[1]
                    self.optimal_vals["p_load_opt"] += self.optimal_vals["p_ev_ch"] + self.optimal_vals["p_ev_v2g"]

                if 'pv' in self.devices:
                    self.pv.resolve()

                if 'battery' in self.devices:
                    self.battery.resolve()


                self.optimal_vals["solve_counter"] = self.counter
                self.optimal_vals["forecast_p_grid_opt"] = self.optimal_vals["p_load_opt"]
                self.optimal_vals["p_grid_opt"] = self.optimal_vals["p_load_opt"]
                self.optimal_vals["cost_opt"] = self.optimal_vals["p_grid_opt"] * self.total_price.value[0]
                
                i+=1

    def set_p_grid(self):
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
        self.current_values = self.redis_client.hgetall("current_values")

    def cast_redis_timestep(self):
        """
        Sets the timestep of the current time with respect to total simulation time.
        
        :return: None
        """
        self.timestep = int(self.current_values["start_index"]) + int(self.current_values["timestep"])

    def cast_redis_curr_rps(self):
        """
        Casts the reward price signal values for the current timestep.
        
        :return: None
        """
        rp = self.redis_client.lrange('reward_price', 0, -1)
        self.reward_price[:self.dt] = float(rp[-1])
        # self.log.debug(f"ts: {self.timestep}; RP: {self.reward_price[0]}")

    def solve_type_problem(self):
        """
        Selects routine for MPC optimization problem setup and solve using home type.
        
        :return: None
        """
        self.add_base_constraints()
        self.set_p_grid()
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

        self.redis_client = rc.connection(self.redis_url)#RedisClient(self.redis_url)
        self.redis_get_initial_values()
        self.cast_redis_timestep()

        if self.timestep > 0:
            self.redis_get_prev_optimal_vals()

        self.get_initial_conditions()
        self.solve_type_problem()
        self.cleanup_and_finish()
        self.redis_write_optimal_vals()

        # self.log.removeHandler(fh)
