import os
import numpy as np
import cvxpy as cp

class EV:
    def __init__(self, hems):
        # set centralized hems for sensor values and occ schedule
        self.hems = hems 

        # create minimum MPC horizon for EV due to long "away" periods
        # (EV should be able to forsee the entirety of its trip away and back + duration)
        self.min_horizon = int(24 * self.hems.dt)
        self.horizon = max(self.hems.horizon, self.min_horizon)
        self.h_plus = self.horizon + 1

        # Define constants
        self.ev_max_rate = 5 # kWh
        self.ev_cap_total = 16 # kWh
        self.ev_cap_min = 0.4*self.ev_cap_total
        self.ev_cap_max = self.ev_cap_total
        self.ev_ch_eff = cp.Constant(0.95)
        self.ev_disch_eff = cp.Constant(0.97)
        self.e_ev_init = cp.Constant(16)
        self.p_ev_disch = cp.Constant(np.zeros(self.horizon)) # uncontrolled discharge while driving

        # Define battery optimization variables
        self.p_ev_ch = cp.Variable(self.horizon) # charge
        self.p_v2g = cp.Variable(self.horizon) # v2g discharge
        self.p_elec = cp.Variable(self.horizon)
        self.e_ev = cp.Variable(self.h_plus)
        self.ev_preference = cp.Constant(np.random.uniform(0.5,0.7))
        self.p_preference = cp.Constant(self.ev_max_rate * 0.3)
        # self.charge_penalty = cp.Variable(1)

        # track these values 
        self.opt_keys = {'p_ev_ch', 'p_ev_disch', 'p_v2g', 'e_ev_opt', 'returning_horizon', 'leaving_horizon'}

    def add_constraints(self, enforce_bounds=True):
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

        self.p_ev_disch = cp.Constant([p_disch_trip/2 if i in [self.hems.today_leaving, self.hems.today_returning, self.hems.tomorrow_leaving, self.hems.tomorrow_returning] else 0 for i in self.hems.subhour_of_day_current[:self.horizon]])
        self.v2g_max = -1 * np.multiply(self.ev_max_rate, self.hems.occ_current[:self.horizon])
        self.p_ch_max = np.minimum(self.p_preference.value * np.ones(self.horizon), np.multiply(self.ev_max_rate, self.hems.occ_current[:self.horizon]))

        cons = [
            self.p_v2g <= 0,
            self.p_v2g >= self.v2g_max,
            self.p_v2g == 0,
            self.e_ev[1:] == self.e_ev[:-1]
                                        + (self.ev_ch_eff * self.p_ev_ch
                                        + self.p_ev_disch / self.ev_disch_eff
                                        + self.p_v2g / self.ev_disch_eff) / self.hems.dt,
            self.e_ev[0] == self.hems.e_ev_init,
            self.p_ev_ch <= self.p_ch_max,
            self.p_ev_ch >= 0,
            self.e_ev >= 0,
            self.e_ev <= self.ev_cap_total,
            self.p_elec == self.p_ev_ch - self.p_v2g,
        ]

        if enforce_bounds:
            cons += [
                self.p_ev_ch <= self.p_preference, # preferred max charge rate
                self.e_ev[1:] <= self.ev_cap_max, # preferred min SOC
                self.e_ev[1:] >= self.ev_cap_min,
            ]

        return cons

    def resolve(self):
        cons = self.add_constraints()
        obj = cp.Minimize(cp.sum(self.p_elec))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=self.hems.solver)

        if not prob.status == "optimal":
            cons = self.add_constraints(enforce_bounds=False)
            obj = cp.Minimize(1)
            prob = cp.Problem(obj, cons)
            prob.solve(solver=self.hems.solver)

    def override_charge(self, cmd):
        self.obj = cp.Variable(1)
        if cmd >= 0:
            cons = [self.obj == cmd * self.ev_max_rate - self.p_ev_ch]
        else:
            cons = [self.obj == cmd * self.ev_max_rate - self.p_v2g]

        prob = cp.Problem(cp.Minimize(self.obj), cons + self.add_constraints())
        prob.solve(solver=self.hems.solver)

        if not prob.status == 'optimal':
            self.resolve()
        return cons

