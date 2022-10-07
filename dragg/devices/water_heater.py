import os
import numpy as np
import cvxpy as cp

class WH:
    """
    A class for the water heater device in a smart home. This model uses a linear R1-C1 model 
    of the water heater tank with electric resistance heating (efficiency ~=100%)
    """
    def __init__(self, hems):
        self.hems = hems
        self.temp_wh_ev = cp.Variable(self.hems.h_plus)
        self.temp_wh = cp.Variable(1)
        self.temp_wh0 = cp.Variable(1)
        self.wh_heat_max = self.hems.sub_subhourly_steps
        self.wh_heat_min = 0

        # Water heater temperature constants
        self.r = cp.Constant(float(self.hems.home["wh"]["r"]) * 1000)
        self.p = cp.Constant(float(self.hems.home["wh"]["p"]) / self.hems.sub_subhourly_steps)
        self.temp_wh_min = cp.Constant(float(self.hems.home["wh"]["temp_wh_min"]))
        self.temp_wh_max = cp.Constant(float(self.hems.home["wh"]["temp_wh_max"]))
        self.temp_wh_sp = cp.Constant(float(self.hems.home["wh"]["temp_wh_sp"]))
        self.wh_size = float(self.hems.home["wh"]["tank_size"])
        self.tap_temp = 15 # assumed cold tap water is about 55 deg F
        wh_capacitance = self.wh_size * 4.2 # kJ/deg C
        self.c = cp.Constant(wh_capacitance)

        self.heat_on = cp.Variable(self.hems.horizon, integer=True)
        self._get_water_draws()
        self.opt_keys = {"waterdraws", "temp_wh_ev_opt", "wh_heat_on_opt"}

        self.override = False

    def _get_water_draws(self):
        """
        :input: None
        :return: None

        A method to determine the current hot water consumption. Data based on NREL database (to cite)
        """
        draw_sizes = (self.hems.horizon // self.hems.dt + 1) * [0] + self.hems.home["wh"]["draw_sizes"]
        raw_draw_size_list = draw_sizes[(self.hems.timestep // self.hems.dt):(self.hems.timestep // self.hems.dt) + (self.hems.horizon // self.hems.dt + 1)]
        raw_draw_size_list = (np.repeat(raw_draw_size_list, self.hems.dt) / self.hems.dt).tolist()
        draw_size_list = raw_draw_size_list[:self.hems.dt]
        for i in range(self.hems.dt, self.hems.h_plus):
            draw_size_list.append(np.average(raw_draw_size_list[i-1:i+2]))

        self.draw_size = draw_size_list
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
            # Hot water heater contraints, expected value after approx waterdraws
            self.temp_wh_ev[0] == self.hems.temp_wh_init,
            self.temp_wh_ev[1:] == (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                + self.draw_frac[1:]*self.tap_temp)
                                + 3600 * ((((self.hems.hvac.temp_in_ev[1:] 
                                    - (cp.multiply(self.remainder_frac[1:],self.temp_wh_ev[:-1]) 
                                    + self.draw_frac[1:]*self.tap_temp)) / self.r))
                                    + self.heat_on * self.p) / (self.c * self.hems.dt),
            self.temp_wh_ev >= self.temp_wh_min,
            self.temp_wh_ev <= self.temp_wh_max,

            self.temp_wh0 == self.hems.temp_wh_init
                            + 3600 * (((self.hems.hvac.temp_in - self.hems.temp_wh_init) / self.r)
                            + self.heat_on[0] * self.p) / (self.c * self.hems.dt),

            self.temp_wh == ((self.temp_wh0 * (self.wh_size - self.draw_size[0])) + (self.tap_temp * self.draw_size[0])) / self.wh_size,

            self.heat_on <= self.wh_heat_max,
            self.heat_on >= self.wh_heat_min]

        if enforce_bounds:
            cons += [
                self.temp_wh >= self.temp_wh_min,
                self.temp_wh <= self.temp_wh_max
                ]

        if self.override:
            cons += [
                self.heat_on[0] <= self.cmd_heat_max,
                self.heat_on[0] >= self.cmd_heat_min
            ]
            self.override=False

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
        obj = cp.Minimize(cp.sum(self.p * self.heat_on))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.GLPK_MI)
        if not prob.status == 'optimal':
            cons = self.add_constraints(enforce_bounds=False)
            obj = cp.Minimize(cp.sum(self.temp_wh_ev - self.temp_wh_max))
            prob = cp.Problem(obj, cons)
            prob.solve(solver=self.hems.solver)

    def override_p_wh(self, cmd):
        """
        :parameter cmd: float in [-1,1]
        :return: None
        
        A method to override the current on/off status of the hot water heater. Directly controls
        the power consumed with a conservative check that the resulting water temperature will not
        exceed bounds in either direction.
        """
        # project the normalized value [-1,1] to [0, self.hems.sub_subhourly_steps]
        self.override = True
        cmd_p = (cmd + 1) * 0.5

        max_heating_deg = self.temp_wh_max.value - self.hems.temp_wh_init.value
        min_heating_deg = self.hems.temp_wh_init.value - self.temp_wh_min.value
        
        approx_max_deg = 3600 * (((self.hems.hvac.t_in_min[0] - self.hems.temp_wh_init.value) / self.r.value)
                            + self.hems.sub_subhourly_steps * self.p.value) / (self.c.value * self.hems.dt)

        max_cmd = max_heating_deg / approx_max_deg
        min_cmd = min_heating_deg / approx_max_deg
        cmd_p = np.clip(cmd_p, min_cmd, max_cmd)
        cmd_p = cmd_p * self.hems.sub_subhourly_steps

        self.cmd_heat_max = cmd_p + 0.5
        self.cmd_heat_min = cmd_p - 0.5