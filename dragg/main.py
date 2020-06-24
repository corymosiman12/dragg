from dragg.aggregator import Aggregator
from dragg.reformat import Reformat
from dragg.logger import Logger
from datetime import datetime
import json
import os
import sys

if __name__ == "__main__":
    # a = Aggregator()
    # a.run()

    agg_params = {"alpha": [0.09, 0.1], "beta":[0.8]} # set parameters from earlier runs
    mpc_params = {}
    include_runs = {"baseline", "rl_agg"}
    r = Reformat(agg_params=agg_params, include_runs=include_runs)

    r.rl2baseline()
    r.rl_thetas()
    r.rl_qvals()
    if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated
        r.plot_single_home2("Myles-XQ5IA") # pv_battery
        # r.plo1t_single_home2(type="base")

        # r.plot_all_homes()
