import os
import numpy as np
import cvxpy as cp

class WH:
    """
    A class for the water heater device in a smart home. This model uses a linear R1-C1 model 
    of the water heater tank with electric resistance heating (efficiency ~=100%)
    """
    def __init__(self, hems):
        # set the centralized hems for current sensor values
        self.hems = hems

        # optimization variables
        self.temp_wh_ev = cp.Variable(self.hems.h_plus)
        self.temp_wh = cp.Variable(1)
        self.temp_wh0 = cp.Variable(1)
        self.heat_on = cp.Variable(self.hems.horizon, integer=True)
        self.p_elec = cp.Variable(self.hems.horizon)
        self.temp_delta = cp.Variable(self.hems.horizon)

        # Water heater temperature constants
        self.r = cp.Constant(float(self.hems.home["wh"]["r"]) * 1000)
        self.p = cp.Constant(float(self.hems.home["wh"]["p"]))
        self.temp_wh_min = cp.Constant(float(self.hems.home["wh"]["temp_wh_min"]))
        self.temp_wh_max = cp.Constant(float(self.hems.home["wh"]["temp_wh_max"]))
        self.temp_wh_sp = cp.Constant(float(self.hems.home["wh"]["temp_wh_sp"]))
        self.wh_size = float(self.hems.home["wh"]["tank_size"]) # in liters
        self.tap_temp = 15 # assumed cold tap water is about 55 deg F
        wh_capacitance = self.wh_size * 4.2 # kJ/deg C
        self.c = cp.Constant(wh_capacitance)
        self.wh_heat_max = self.hems.sub_subhourly_steps # integer represents duty cycle
        self.wh_heat_min = 0
        
        # initilize water draws
        self._get_water_draws()
        self.opt_keys = {"waterdraws", "temp_wh_opt", "wh_heat_on_opt"}

    def _get_water_draws(self):
        """
        :input: None
        :return: None

        A method to determine the current hot water consumption. Data based on NREL database (to cite)
        """
        daily_draw_sizes = 2 * self.hems.home["wh"]["draw_sizes"] # repeat list for 2 weeks

        current_ts = [i % (24 * 7 * self.hems.dt) for i in range(self.hems.timestep, self.hems.timestep + self.hems.dt)]
        current_draw_size_actual = [daily_draw_sizes[i] for i in current_ts]
        future_ts = [i % (24 * 7 * self.hems.dt) for i in range(self.hems.timestep + self.hems.dt, self.hems.timestep + self.hems.h_plus)]
        future_draw_size_ev = [np.average(daily_draw_sizes[i-self.hems.dt:i+self.hems.dt]) for i in future_ts]

        self.draw_size = current_draw_size_actual + future_draw_size_ev
        df = np.divide(self.draw_size, self.wh_size)
        self.draw_frac = cp.Constant(df)
        self.remainder_frac = cp.Constant(1-df)
        return

    def add_constraints(self, enforce_bounds=True):
        """
        :parameter enforce_bounds: boolean determines whether comfort bounds are strictly enforced
        :return: cons, a list of CVXPY constraints

        A method to introduce physical constraints to the water heater. 
        """
        self._get_water_draws()

        cons = [
            # thermal constraints, expected value after approx waterdraws
            self.temp_wh_ev[0] == self.hems.temp_wh_init,
            self.temp_wh_ev[1:] == (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                + self.draw_frac[1:]*self.tap_temp) + self.temp_delta,

            self.temp_delta == (3600 * self.hems.dt_frac) * ((((self.hems.hvac.temp_in_ev[1:] 
                                    - (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                    + self.draw_frac[1:]*self.tap_temp)) / self.r))
                                    + self.heat_on * self.p) / (self.c),

            # actual value of temperature of tank at time t="-0.5", before any water draw occurs
            self.temp_wh0 == self.hems.temp_wh_init
                            + (3600 * self.hems.dt_frac) * (((self.hems.hvac.temp_in - self.hems.temp_wh_init) / self.r)
                            + self.heat_on[0] * self.p) / (self.c),

            # actual value of temperature of tank at time t=0, after any water draw occurs, before heating
            # self.temp_wh == ((self.temp_wh0 * self.remainder_frac[0]) + (self.tap_temp * self.draw_frac[0])), #/ self.wh_size,
            self.temp_wh == self.temp_wh_ev[1],

            # electrical power as a function of thermal power
            self.heat_on <= self.wh_heat_max,
            self.heat_on >= self.wh_heat_min,
            self.p_elec == (self.heat_on * self.p) / self.hems.sub_subhourly_steps # kW
            ]

        if enforce_bounds:
            # optional comfort constraints
            cons += [
                    self.temp_wh >= self.temp_wh_min,
                    self.temp_wh <= self.temp_wh_max,
                    self.temp_wh_ev >= self.temp_wh_min,
                    self.temp_wh_ev <= self.temp_wh_max,
                ]

        return cons 

    def resolve(self):
        """
        :input: None
        :return: None

        A method for re-solving the constraints specific to the hot water heater in the event
        that the whole house HEMS cannot satisfy all constraints -- first attempts to minimize the 
        device-specific electricity consumption while satisfying comfort bounds, second attempt 
        minimizes the deviation of the new temperature and the desired setpoint.
        """
        cons = self.add_constraints()
        obj = cp.Minimize(cp.abs(self.temp_wh - self.temp_wh_min))
        # obj = cp.Maximize(self.temp_wh)
        prob = cp.Problem(obj, cons)
        prob.solve(solver=self.hems.solver)
        if not prob.status == 'optimal':
            cons = self.add_constraints(enforce_bounds=False)
            obj = cp.Minimize(self.temp_wh_min - self.temp_wh)
            prob = cp.Problem(obj, cons)
            prob.solve(solver=self.hems.solver, verbose=False)
            print(prob.status)

    def add_override_cons(self):
        """
        :parameter enforce_bounds: boolean determines whether comfort bounds are strictly enforced
        :return: cons, a list of CVXPY constraints

        A method to introduce physical constraints to the water heater. 
        """
        self._get_water_draws()

        cons = [
            # thermal constraints, expected value after approx waterdraws
            self.temp_wh_ev[0] == self.hems.temp_wh_init,
            self.temp_wh_ev[1:] == (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                + self.draw_frac[1:]*self.tap_temp) + self.temp_delta,

            self.temp_delta == (3600 * self.hems.dt_frac) * ((((self.copy_hvac 
                                    - (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                    + self.draw_frac[1:]*self.tap_temp)) / self.r))
                                    + self.heat_on * self.p) / (self.c),

            # actual value of temperature of tank at time t="-0.5", before any water draw occurs
            self.temp_wh0 == self.hems.temp_wh_init
                            + (3600 * self.hems.dt_frac) * (((self.copy_hvac[0] - self.hems.temp_wh_init) / self.r)
                            + self.heat_on[0] * self.p) / (self.c),

            # actual value of temperature of tank at time t=0, after any water draw occurs, before heating
            # self.temp_wh == ((self.temp_wh0 * self.remainder_frac[0]) + (self.tap_temp * self.draw_frac[0])), #/ self.wh_size,
            self.temp_wh == self.temp_wh_ev[1],

            # electrical power as a function of thermal power
            self.heat_on <= self.wh_heat_max,
            self.heat_on >= self.wh_heat_min,
            self.p_elec == (self.heat_on * self.p) / self.hems.sub_subhourly_steps, # kW

            self.temp_wh >= self.temp_wh_min,
            self.temp_wh <= self.temp_wh_max,
            self.temp_wh_ev >= self.temp_wh_min,
            self.temp_wh_ev <= self.temp_wh_max,
            ]

        return cons 

    def override_p_wh(self, cmd):
        """
        :parameter cmd: float in [-1,1]
        :return: None
        
        A method to override the current on/off status of the hot water heater. Directly controls
        the power consumed with a conservative check that the resulting water temperature will not
        exceed bounds in either direction.
        """
        self.obj = cp.Variable(1)
        self.copy_hvac = self.hems.hvac.temp_in_ev[1:].value
        cons = self.add_override_cons()

        cmd = (0.5 * cmd + 0.5)
        obj_cons = [self.obj == (cmd - (self.heat_on[0] / self.wh_heat_max)), self.obj >= 0]
        obj = cp.Minimize(self.obj)
        prob = cp.Problem(obj, cons + obj_cons)
        prob.solve(solver=self.hems.solver)
        
        if not prob.status == 'optimal':
            obj_cons = [self.obj == (self.heat_on[0] / self.wh_heat_max) - (0.5 * cmd + 0.5)]
            prob = cp.Problem(obj, cons + obj_cons)
            prob.solve(solver=self.hems.solver)

            if not prob.status == 'optimal':
                print('resolve')
                self.resolve()

        return obj_cons 