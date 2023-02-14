import os
import numpy as np
import cvxpy as cp

class HVAC:
    """
    A class for the HVAC device in a smart home. This model uses a linear R1-C1 thermal model 
    to determine the indoor air temperature of the home. 

    Source of linear model: https://hal.archives-ouvertes.fr/hal-01739625/document 
    """
    def __init__(self, hems):
        # set the centralized hems
        self.hems = hems

        # set hvac cooling and heating parameters, random dist from aggregator
        self.p_c = cp.Constant(float(self.hems.home["hvac"]["p_c"])) # thermal power (kW)
        self.p_h = cp.Constant((float(self.hems.home["hvac"]["p_h"])))
        self.cop_c = 0.293 * cp.Constant(float(self.hems.home["hvac"]["hvac_seer"])) # seer = thermal power / electrical power (kBtu/kW)
        self.cop_h = 0.293 * cp.Constant(float(self.hems.home["hvac"]["hvac_hspf"]))

        # coefficient matrix
        self.b = [1/(self.hems.r.value*self.hems.c.value), self.hems.window_eq.value/self.hems.c.value, 1/self.hems.c.value]

        # optimization variables
        self.p_elec = cp.Variable(self.hems.horizon)
        self.temp_in_ev = cp.Variable(self.hems.h_plus)
        self.temp_in = cp.Variable(1)
        self.cool_on = cp.Variable(self.hems.horizon, integer=True) # integer represents duty cycle
        self.heat_on = cp.Variable(self.hems.horizon, integer=True)

        # Home temperature constraints
        # create temp bounds based on occ schedule, array should be one week long
        self.occ_t_in_min = float(self.hems.home["hvac"]["temp_in_min"])
        self.unocc_t_in_min = float(self.hems.home["hvac"]["temp_in_min"]) - float(self.hems.home["hvac"]["temp_setback_delta"])
        self.occ_t_in_max = float(self.hems.home["hvac"]["temp_in_max"])
        self.unocc_t_in_max = float(self.hems.home["hvac"]["temp_in_max"]) + float(self.hems.home["hvac"]["temp_setback_delta"])
        self.t_deadband = self.occ_t_in_max - self.occ_t_in_min
        
        self.opt_keys = {"temp_in_opt","hvac_cool_on_opt", "hvac_heat_on_opt","t_in_min", "t_in_max", "occupancy_status"}

    def add_constraints(self, enforce_bounds=[True, True]):
        """
        :parameter enforce_bounds: boolean determines whether comfort bounds are strictly enforced
        :return: cons, a list of CVXPY constraints

        A method to introduce physical constraints to the HVAC equipment. The A/C and heat are 
        alternately disabled by "season" to reduce on/off cycling and/or simaultaneous heating
        and cooling when the electricity price is negative.
        """

        self.t_in_min_current = cp.Constant([self.occ_t_in_min if i else self.unocc_t_in_min for i in self.hems.occ_current[:self.hems.h_plus]])
        self.t_in_max_current = cp.Constant([self.occ_t_in_max if i else self.unocc_t_in_max for i in self.hems.occ_current[:self.hems.h_plus]])

        self.cool_min = 0 
        self.heat_min = 0
        self.heat_max = self.hems.sub_subhourly_steps 
        self.cool_max = self.hems.sub_subhourly_steps 

        cons = [
            # Physical indoor air temperature constraints
            # self.temp_in_ev = indoor air temperature expected value (prediction)
            self.temp_in_ev[0] == self.hems.temp_in_init,
            self.temp_in_ev[1:] == self.temp_in_ev[:-1]
                                + (self.b[0] * (self.hems.oat_current[1:] - self.temp_in_ev[:-1]) 
                                + self.b[1] * self.hems.ghi_current[1:] 
                                - self.b[2] * self.cool_on * self.p_c * 1000
                                + self.b[2] * self.heat_on * self.p_h * 1000) * 3600 * self.hems.dt_frac,
            
            # electric power as a function of thermal power
            # note: hems.dt_frac = fraction of an hour that corresponds to minimum duty cycle runtime 
            self.p_elec == ((self.cool_on * self.p_c / self.cop_c) + (self.heat_on * self.p_h / self.cop_h)) / self.hems.sub_subhourly_steps,

            # temperature at the next timestep (actual value)
            self.temp_in == self.hems.temp_in_init
                            + (self.b[0] * (self.hems.oat_current[1] - self.hems.temp_in_init) 
                            + self.b[1] * self.hems.ghi_current[1]
                            - self.b[2] * self.cool_on[0] * self.p_c * 1000
                            + self.b[2] * self.heat_on[0] * self.p_h * 1000) * 3600 * self.hems.dt_frac,

            # constrain cooling on/off to be integer (duty cycle)
            self.cool_on <= self.cool_max,
            self.cool_on >= self.cool_min,
            self.heat_on <= self.heat_max,
            self.heat_on >= self.heat_min,
            ]

        if enforce_bounds[0]:
            # Enforces comfort constraints, which are sometimes impossible to adhere to
            cons += [
                    self.temp_in <= self.t_in_max_current[0],
                    self.temp_in >= self.t_in_min_current[0],
                    self.temp_in_ev[1:] >= self.t_in_min_current[:self.hems.horizon],
                    self.temp_in_ev[1:] <= self.t_in_max_current[:self.hems.horizon],
                ]

        if enforce_bounds[1]:
            # safety bounds for temperature -- always obeyed
            self.temp_in_ev >= self.unocc_t_in_min,
            self.temp_in_ev <= self.unocc_t_in_max,

        return cons

    def resolve(self):
        """
        :return: none

        Re-solves only the HVAC portion of the MPC scheduling problem, since sometimes the comfort
        constraints are impossible to adhere to the comfort bounds are not enforced but the difference
        in the observed temp and the desired temp is minimized.
        """
        # add physical constraints, attempt to obey thermal preference
        cons = self.add_constraints()

        # not the ideal solution to resolve heat and cooling at the same time
        if self.hems.temp_in_init.value <= self.t_in_max_current[0].value:
            cons += [self.cool_on == 0]
        if self.hems.temp_in_init.value >= self.t_in_min_current[0].value:
            cons += [self.heat_on == 0]

        obj = cp.Minimize(self.p_elec[0])
        prob = cp.Problem(obj, cons)
        prob.solve(solver=self.hems.solver)

        if not prob.status == 'optimal':
            # if thermal preferences are infeasible resolve within safety bounds
            cons = self.add_constraints(enforce_bounds=[False, True])
            
            # not the ideal solution to resolve heat and cooling at the same time
            if self.hems.temp_in_init.value <= self.t_in_max_current[0].value:
                cons += [self.cool_on == 0]
            if self.hems.temp_in_init.value >= self.t_in_min_current[0].value:
                cons += [self.heat_on == 0]

            obj = cp.Minimize(cp.sum(cp.abs(self.temp_in_ev - (self.t_in_min_current + 0.5 * self.t_deadband))))
            prob = cp.Problem(obj, cons)
            prob.solve(solver=self.hems.solver)
            # print(prob.status)

            if not prob.status == 'optimal':
                # if thermal preferences are infeasible resolve within safety bounds
                cons = self.add_constraints(enforce_bounds=[False, False])
                
                # not the ideal solution to resolve heat and cooling at the same time
                if self.hems.temp_in_init.value <= self.t_in_max_current[0].value:
                    cons += [self.cool_on == 0]
                if self.hems.temp_in_init.value >= self.t_in_min_current[0].value:
                    cons += [self.heat_on == 0]

                obj = cp.Minimize(cp.sum(cp.abs(self.temp_in_ev - (self.t_in_min_current + 0.5 * self.t_deadband))))
                prob = cp.Problem(obj, cons)
                prob.solve(solver=self.hems.solver)
                # print(prob.status)

    def override_t_in(self, cmd):
        """
        :parameter cmd: float between [-1,1]
        :return: none

        A method for manually setting the temperature setpoint in the home.
        The result will set the thermal setpoint between the min and max safety bounds (unoccupied 
        temperatures) as dictated by the normalized command.
        """
        t_sp = cmd * self.t_deadband + self.occ_t_in_min
        self.obj = cp.Variable(1)
        if self.hems.temp_in_init.value > t_sp:
            cons = [
                self.obj == t_sp - self.temp_in,
                self.cool_on == 0
            ]
        else:
            cons = [
                self.obj == self.temp_in - t_sp,
                self.heat_on == 0
            ]
        prob = cp.Problem(cp.Minimize(self.obj), cons + self.add_constraints())
        prob.solve(solver=self.hems.solver)
        if not prob.status == 'optimal':
            self.resolve()
        return cons


