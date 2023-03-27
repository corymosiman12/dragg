import os
import numpy as np
import cvxpy as cp

class Battery:
    """
    A class for the HVAC device in a smart home. This model uses a linear R1-C1 thermal model 
    to determine the indoor air temperature of the home. 

    Source of linear model: https://hal.archives-ouvertes.fr/hal-01739625/document 
    """
    def __init__(self, hems):
        self.hems = hems
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
        self.p_elec = cp.Variable(self.horizon)
        self.e_batt = cp.Variable(self.h_plus)

        self.opt_keys = {'p_batt_ch', 'p_batt_disch', 'e_batt_opt'}

    def add_constraints(self):
        cons = [
        # Battery constraints
            self.e_batt[1:] == self.e_batt[:-1]
                + (self.batt_ch_eff * self.p_batt_ch 
                               + self.p_batt_disch / self.batt_disch_eff) / self.hems.dt,
            self.e_batt[0] == self.hems.e_batt_init,
            self.p_batt_ch <= self.batt_max_rate,
            -self.p_batt_disch <= self.batt_max_rate,
            self.p_batt_ch >= 0,
            self.p_batt_disch <= 0,
            self.p_elec == self.p_batt_ch - self.p_batt_disch,
            self.e_batt[1:] <= self.batt_cap_max,
            self.e_batt[1:] >= self.batt_cap_min]
        return cons

    def resolve(self):
        cons = self.add_constraints()
        obj = cp.Minimize(cp.sum(self.p_batt_ch))
        prob = cp.Problem(obj, cons)
        prob.solve()