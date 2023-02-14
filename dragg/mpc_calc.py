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

import faulthandler; faulthandler.enable()

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

        self.setup_base_problem()
        self.optimal_vals = {k:0 for k in self.opt_keys}

        self.ev_override_profile = None

        offset = -3 if self.home['hems']['schedule_group'] == 'early_birds' else 3 if self.home['hems']['schedule_group'] == 'night_owls' else 0
        # self.typ_leave = np.random.randint(8,10) + offset
        # self.typ_return = np.random.randint(18,20) + offset

        self.tomorrow_leaving = self.typ_leave + (24*self.dt)
        self.tomorrow_returning = self.typ_return + (24*self.dt)
        self.today_leaving = self.typ_leave
        self.today_returning = self.typ_return

        self.redis_client.hset(self.name, "today_leaving", self.today_leaving)
        self.redis_client.hset(self.name, "today_returning", self.today_returning)
        self.redis_client.hset(self.name, "tomorrow_leaving", self.tomorrow_leaving)
        self.redis_client.hset(self.name, "tomorrow_returning", self.tomorrow_returning)

        self.late_night_return = 0
        self.leaving_index = -1
        self.returning_index = -1

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
                print("failed to post", key, field, value)

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
        self.start_hour_index = self.redis_client.hget('current_values','start_index')
        self.all_weekday = self.redis_client.lrange('WEEKDAY', 0, -1)
        self.all_hours = self.redis_client.lrange('Hour', 0, -1)
        self.all_months = self.redis_client.lrange('Month', 0, -1)
        self.all_ghi = self.redis_client.lrange('GHI', 0, -1)
        self.all_oat = self.redis_client.lrange('OAT', 0, -1)
        self.all_spp = self.redis_client.lrange('SPP', 0, -1)
        self.all_tou = self.redis_client.lrange('tou', 0, -1)
        self.base_cents = float(self.all_tou[0])

        # cast all values to proper type
        self.start_hour_index = int(float(self.start_hour_index))
        self.start_slice = self.start_hour_index
        self.all_weekday = [float(i) for i in self.all_weekday]
        self.all_hours = [float(i) for i in self.all_hours]
        self.all_months = [float(i) for i in self.all_months]
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

        self.solver = self.home['hems']['solver']

        # Set up the horizon for the MPC calc (min horizon = 1, no MPC)
        self.sub_subhourly_steps = max(1, int(self.home['hems']['sub_subhourly_steps']))
        self.dt = max(1, int(self.home['hems']['hourly_agg_steps']))
        self.horizon = max(1, int(self.home['hems']['horizon']) * int(self.dt))
        self.h_plus = self.horizon + 1
        self.discount = float(self.home['hems']['discount_factor'])
        self.dt_frac = 1 / self.dt / self.sub_subhourly_steps

        self.typ_leave = float(self.home['hems']['typ_leave'])
        self.typ_return = float(self.home['hems']['typ_return'])
        
        # Initialize RP structure so that non-forecasted RPs have an expected value of 0.
        self.reward_price = np.zeros(self.horizon)

        # Define optimization variables
        self.p_load = cp.Variable(self.horizon)
        self.p_grid = cp.Variable(self.horizon)

        self.r = cp.Constant(float(self.home["hvac"]["r"]))
        self.c = cp.Constant(float(self.home["hvac"]["c"])* 1e7) 
        self.window_eq = cp.Constant(float(self.home["hvac"]["w"]))

        self.max_load = 0
        if 'hvac' in self.devices:
            self._setup_hvac_problem()
            self.max_load += max(self.hvac.p_c.value, self.hvac.p_h.value)
        if 'wh' in self.devices:
            self._setup_wh_problem()
            self.max_load += self.wh.p.value
        if 'ev' in self.devices:
            self._setup_ev_problem()
            self.max_load += self.ev.ev_max_rate
        if 'battery' in self.devices:
            self._setup_battery_problem()
            self.max_load += self.battery.batt_max_rate
        if 'pv' in self.devices:
            self._setup_pv_problem()

        self.opt_keys.update({"p_grid_opt", "forecast_p_grid_opt", "p_load_opt", "cost_opt", "occupancy_status"})
    
    def get_daily_occ(self):
        """
        Get the daily occupancy schedule of the day and tomorrow.
        """
        if False: # self.subhour_of_day_current[0] == 0 and not self.name=="PLAYER":
            self.today_leaving = self.tomorrow_leaving % 24
            self.today_returning = self.tomorrow_returning % 24
            if self.weekday_current[0] in [6,0,1,2,3]: # if the next day is a weekday
                self.tomorrow_leaving = int(self.typ_leave + np.random.normal() + 24)
                self.tomorrow_returning = int(self.tomorrow_leaving + np.random.randint(6,9))
            else:
                self.tomorrow_leaving = int(np.random.randint(24,49))
                self.tomorrow_returning = int(self.tomorrow_leaving + np.random.randint(0,6))
            self.redis_client.hset(self.name, "today_leaving", self.today_leaving)
            self.redis_client.hset(self.name, "today_returning", self.today_returning)
            self.redis_client.hset(self.name, "tomorrow_leaving", self.tomorrow_leaving)
            self.redis_client.hset(self.name, "tomorrow_returning", self.tomorrow_returning)

        else:
            self.today_leaving = float(self.redis_client.hget(self.name, "today_leaving"))
            self.today_returning = float(self.redis_client.hget(self.name, "today_returning"))
            self.tomorrow_leaving = float(self.redis_client.hget(self.name, "tomorrow_leaving"))
            self.tomorrow_returning = float(self.redis_client.hget(self.name, "tomorrow_returning"))
        
        self.occ_current = [0 if (self.today_leaving <= x <= self.today_returning) or (self.tomorrow_returning <= x <= self.tomorrow_leaving) else 1 for x in self.subhour_of_day_current]
        self.leaving_index = [np.argmin(np.abs(np.subtract(self.subhour_of_day_current, time))) for time in [self.today_leaving, self.tomorrow_leaving]]
        self.returning_index = [np.argmin(np.abs(np.subtract(self.subhour_of_day_current, time))) for time in [self.today_returning, self.tomorrow_returning]]
        return 

    def set_environmental_variables(self):
        """
        Slices cast values of the environmental values for the current timestep.
        
        :return: None
        """
        self.start_slice = self.timestep
        end_slice = self.start_slice + self.h_plus # Need to extend 1 timestep past horizon for OAT slice

        # Get current occupancy schedule
        self.weekday_current = self.all_weekday[self.start_slice:end_slice]
        # self.subhour_of_day_current = self.all_hours[self.start_slice:self.start_slice + (24*self.dt) + 1]
        self.subhour_of_day_current = np.linspace(self.all_hours[self.start_slice], self.all_hours[self.start_slice] + 24, (24*self.dt) +1).tolist()
        self.get_daily_occ()

        # Get the current values from a list of all values
        self.ghi_current = cp.Constant(self.all_ghi[self.start_slice:end_slice])
        self.ghi_current_ev = deepcopy(self.ghi_current.value)
        ghi_noise = np.multiply(self.ghi_current.value[1:], 0.01 * np.power(1.3*np.ones(self.horizon), np.arange(self.horizon)))
        self.ghi_current_ev[1:] = np.add(self.ghi_current.value[1:], ghi_noise)

        self.oat_current = cp.Constant(self.all_oat[self.start_slice:end_slice])
        self.oat_current_ev = deepcopy(self.oat_current.value)
        oat_noise = np.multiply(np.power(1.1*np.ones(self.horizon), np.arange(self.horizon)), np.random.randn(self.horizon))
        self.oat_current_ev[1:] = np.add(self.oat_current.value[1:], oat_noise)

        self.tou_current = self.all_tou[self.start_slice:end_slice]
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
        """
        Adds CVX variables for EV subsystem in base homes.
        
        :return: None
        """
        self.ev = EV(self)
        self.opt_keys.update(self.ev.opt_keys)

    def _setup_pv_problem(self):
        """
        Adds CVX variables for photovoltaic subsystem in pv and battery_pv homes.
        
        :return: None
        """
        self.pv = PV(self)
        self.opt_keys.update(self.pv.opt_keys)

    def get_initial_conditions(self):
        if int(self.current_values["timestep"])== 0:
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

    def add_base_constraints(self):
        """
        Creates the system dynamics for thermal energy storage systems: HVAC and
        water heater.
        
        :return: None
        """
        self.constraints = [
            self.p_load == (self.hvac.p_elec + self.wh.p_elec + self.ev.p_elec[:self.horizon]) #/ self.sub_subhourly_steps
        ]
        if 'hvac' in self.devices:
            self.constraints += self.hvac.add_constraints()
        if 'wh' in self.devices:
            self.constraints += self.wh.add_constraints()
        if 'ev' in self.devices:
            self.constraints += self.ev.add_constraints()
        if 'pv' in self.devices:
            self.constraints += self.pv.add_constraints()
        if 'battery' in self.devices:
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
            self.p_grid == self.p_load + (self.battery.p_elec)
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
            self.p_grid == self.p_load - self.pv.p_elec 
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
            self.p_grid == self.p_load + (self.battery.p_elec - self.pv.p_elec)
        ]

    def solve_mpc(self, debug=False):
        """
        Sets the objective function of the Home Energy Management System to be the
        minimization of cost over the MPC time horizon and solves via CVXPY.
        Used for all home types.
        
        :return: None
        """
        self.cost = cp.Variable(self.horizon)
        self.constraints += [self.cost == cp.multiply(self.total_price, self.p_grid / self.dt)] # think this should work
        self.weights = cp.Constant(np.power(self.discount*np.ones(self.horizon), np.arange(self.horizon)))
        self.obj = cp.Minimize(cp.sum(cp.multiply(self.cost, self.weights))) 
        self.prob = cp.Problem(self.obj, self.constraints)
        if not self.prob.is_dcp():
            # self.log.error("Problem is not DCP")
            print("problem is not dcp")
        self.prob.solve(solver=self.solver, verbose=self.verbose_flag)
        return

    def solve_local_control(self):
        """
        Solves the MPC as if each home has its own objective to meet a setpoint (not cost minimizing).
        Can be used in place of .solve_mpc()

        :return: None
        """
        # self.obj = cp.Minimize(self.ev.obj + self.hvac.obj + self.wh.obj)

        cons = [
            self.p_load == (self.hvac.p_elec.value + self.wh.p_elec.value + self.ev.p_elec[:self.horizon].value), #/ self.sub_subhourly_steps
            self.p_grid == self.p_load,
            self.cost == cp.multiply(self.total_price, self.p_grid / self.dt)
        ]

        self.prob = cp.Problem(cp.Minimize(1), cons)
        self.prob.solve(solver=self.solver)
        return 

    def cleanup_and_finish(self):
        """
        Resolves .solve_mpc() with error handling and collects all data on solver.
        
        :return: None
        """
        # end_slice = max(1, self.sub_subhourly_steps)
        
        i = 0
        while i < 1:
            if self.prob.status == 'optimal': # if the problem has been solved
                self.counter = 0
                self.stored_optimal_vals = defaultdict()

                # general base values
                self.stored_optimal_vals["cost_opt"] = (self.cost.value).tolist()
                self.stored_optimal_vals["p_grid_opt"] = (self.p_grid.value / 1).tolist()
                self.stored_optimal_vals["forecast_p_grid_opt"] = (self.p_grid.value[1:] / 1).tolist() + [0]
                self.stored_optimal_vals["p_load_opt"] = (self.p_load.value / 1).tolist()
                self.stored_optimal_vals["occupancy_status"] = [int(x) for x in self.occ_current]
                
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
                    self.stored_optimal_vals['leaving_horizon'] = [int(x) for x in self.leaving_index]
                    self.stored_optimal_vals['returning_horizon'] = [int(x) for x in self.returning_index]

                if 'pv' in self.type:
                    self.stored_optimal_vals['p_pv_opt'] = (self.pv.p_elec.value).tolist()
                    self.stored_optimal_vals['u_pv_curt_opt'] = (self.pv.u.value).tolist()

                if 'battery' in self.type:
                    self.stored_optimal_vals['e_batt_opt'] = (self.battery.e_batt.value).tolist()[1:]
                    self.stored_optimal_vals['p_batt_ch'] = (self.battery.p_batt_ch.value).tolist()
                    self.stored_optimal_vals['p_batt_disch'] = (self.battery.p_batt_disch.value).tolist()
                
                for k in self.opt_keys:
                    self.optimal_vals[k] = self.stored_optimal_vals[k][0]
                    if not k in ['leaving_horizon', 'returning_horizon']:
                        for j in range(self.horizon):
                            try:
                                self.optimal_vals[f"{k}_{j} "] = self.stored_optimal_vals[k][j]
                            except:
                                pass

                if 'hvac' in self.devices:
                    self.optimal_vals["temp_in_opt"] = self.stored_optimal_vals["temp_in_opt"][0]

                if 'wh' in self.devices:
                    self.optimal_vals["temp_wh_opt"] = self.stored_optimal_vals["temp_wh_opt"][0]
                
                self.optimal_vals["correct_solve"] = 1
                self.optimal_vals["solve_counter"] = 0
                # self.log.debug(f"MPC solved with status {self.prob.status} for {self.name}")
                return
            
            else: # clean up solutions if the preference constraints are infeasible
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
                    self.optimal_vals['leaving_horizon'] = int(self.leaving_index[0])
                    self.optimal_vals['returning_horizon'] = int(self.returning_index[0])

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
        """
        Sets the constraint for the total grid power (load + any auxiliary subsystems)

        :return: None
        """

        if self.type == "base":
            self.set_base_p_grid()
        elif self.type == "pv_only":
            self.set_pv_only_p_grid()
        elif self.type == "battery_only":
            self.set_battery_only_p_grid()
        else:
            self.set_pv_battery_p_grid()
        return 

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

        self.redis_client = rc.connection(self.redis_url)
        self.redis_get_initial_values()
        self.cast_redis_timestep()


        if self.timestep > 0:
            self.redis_get_prev_optimal_vals()

        self.get_initial_conditions()
        self.solve_type_problem()
        self.cleanup_and_finish()
        self.redis_write_optimal_vals()