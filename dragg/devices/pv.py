import os
import numpy as np
import cvxpy as cp

class PV:
    """
    A class for the HVAC device in a smart home. This model uses a linear R1-C1 thermal model 
    to determine the indoor air temperature of the home. 

    Source of linear model: https://hal.archives-ouvertes.fr/hal-01739625/document 
    """
    def __init__(self, hems):
        self.hems = hems
        # Define constants
        self.area = cp.Constant(float(self.hems.home["pv"]["area"]))
        self.eff = cp.Constant(float(self.hems.home["pv"]["eff"]))

        # Define PV Optimization variables
        self.p_elec = cp.Variable(self.hems.horizon)
        self.u = cp.Variable(self.hems.horizon)

        self.opt_keys = {'p_pv_opt', 'u_pv_curt_opt'}

    def add_constraints(self):
        cons = [
            # PV constraints.  GHI provided in W/m2 - convert to kWh
            self.p_elec == self.area * self.eff * cp.multiply(self.hems.ghi_current[1:], (1 - self.u)) / 1000,
            self.u >= 0,
            self.u <= 1
        ]
        return cons

    def resolve(self):
        cons = self.add_constraints()
        obj = cp.Maximize(cp.sum(self.p_elec))
        prob = cp.Problem(obj, cons)
        prob.solve()