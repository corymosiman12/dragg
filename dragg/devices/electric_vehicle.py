import os
import numpy as np
import cvxpy as cp

class EV:
    def __init__(self, hems):
        self.hems = hems 

        # Define constants
        self.ev_max_rate = 5 # kWh
        self.ev_cap_total = 16 # kWh
        self.ev_cap_min = 0.2*self.ev_cap_total
        self.ev_cap_max = self.ev_cap_total
        self.ev_ch_eff = cp.Constant(0.95)
        self.ev_disch_eff = cp.Constant(0.97)

        # Define battery optimization variables
        self.p_ev_ch = cp.Variable(self.hems.horizon)
        self.p_ev_disch = cp.Constant(self.hems.horizon)
        self.p_v2g = cp.Variable(self.hems.horizon)
        self.e_ev = cp.Variable(self.hems.h_plus)

        self.e_ev_init = cp.Constant(16)
        self.ev_override_profile = None

        self.opt_keys = {'p_ev_ch', 'p_ev_disch', 'p_v2g', 'e_ev_opt'}

        self.override = False

    def add_constraints(self):
        """
        Creates constraints that make the battery act as an EV with charge/discharge constraints
        based on occupancy and travel distance.
        
        :return: None
        """
        trip_mi = 41 # round trip avg (home to work)
        full_charge = 150 # low end for nissan leaf
        min_daily_soc = 41/150
        ev_batt_cap = 16 # kWh
        e_disch_trip = min_daily_soc * ev_batt_cap
        p_disch_trip = -1 * e_disch_trip * self.hems.dt 

        start = (self.hems.timestep) 
        end = (start + self.hems.h_plus) 
        index = [i % (24 * self.hems.dt) for i in range(start, end)]
        self.index_8am = [i for i, e in enumerate(index) if e in self.hems.leaving_times]
        self.index_5pm = [i-1 for i, e in enumerate(index) if e in self.hems.returning_times]
        after_5pm = [i for i, e in enumerate(index) if e in self.hems.returning_times]
        self.occ_slice = [self.hems.occ_on[i] for i in index]

        self.e_ev_min = []
        for i in range(1,self.hems.h_plus):
            if i in self.index_8am:
                self.e_ev_min += [(e_disch_trip + self.ev_cap_min)]
                # self.constraints += [self.e_ev[i] >= (e_disch_trip + self.ev_cap_min)]
            elif i in after_5pm:
                self.e_ev_min += [0]
                # self.constraints += [self.e_ev[i] >= 0]
            else:
                self.e_ev_min += [self.ev_cap_min]
                # self.constraints += [self.e_ev[i] >= self.ev_cap_min]
        
        self.p_ev_disch = cp.Constant([p_disch_trip if i in self.index_5pm else 0 for i in range(self.hems.horizon)])

        cons = [
            self.p_v2g <= 0,
            self.p_v2g >= -1 * np.multiply(self.ev_max_rate, self.occ_slice)[:-1],
            self.e_ev[1:] == self.e_ev[:-1]
                                        + (self.ev_ch_eff * self.p_ev_ch
                                        + self.p_ev_disch / self.ev_disch_eff
                                        + self.p_v2g / self.ev_disch_eff) / self.hems.dt,
            self.e_ev[0] == self.hems.e_ev_init,
            self.p_ev_ch <= self.ev_max_rate,
            self.p_ev_ch >= 0,
            self.e_ev[1:] <= self.ev_cap_max,
            self.e_ev[1:] >= self.e_ev_min
        ]

        if self.override:
            cons += [
                self.p_ev_ch[0] == self.pc_cmd,
                self.p_ev_dis[0] == self.pd_cmd
            ]

        return cons

    def resolve(self):
        cons = [
            self.e_ev[1:self.hems.h_plus] == self.e_ev[:-1]
                                        + self.p_ev_disch / self.ev_disch_eff / self.hems.dt,
            self.e_ev[0] == self.e_ev_init,
            self.p_ev_ch >= 0,
            self.e_ev[1:] <= self.ev_cap_max,
            self.p_v2g >= -1 * np.multiply(self.ev_max_rate, self.occ_slice[:-1])
        ]
        for i in range(len(self.occ_slice[:-1])):
            if self.occ_slice[i] == 0:
                cons += [self.p_ev_ch[i] == 0]
        if self.ev_override_profile:
            ev_obj = cp.Minimize(cp.sum(self.e_ev - self.ev_override_profile))
        else:
            ev_obj = cp.Minimize(0)
        ev_prob = cp.Problem(ev_obj, cons)
        ev_prob.solve(solver=self.hems.solver)

    def override_charge(self, cmd):
        max_p_ch = (self.ev_cap_max - self.hems.e_ev_init.value) / self.ev_ch_eff.value
        max_p_dis = (self.hems.e_ev_init.value - self.ev_cap_min) * self.ev_disch_eff.value
        self.pc_cmd = np.clip(cmd, 0, max_p_ch)
        self.pd_cmd = -1 * np.clip(cmd, 0, max_p_dis)
