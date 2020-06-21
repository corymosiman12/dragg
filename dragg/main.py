from dragg.aggregator import Aggregator
from dragg.reformat import Reformat
from dragg.logger import Logger
from datetime import datetime
import json
import os
import sys

if __name__ == "__main__":
    a = Aggregator()
    a.run()

    agg_params = {"alpha": [0.78]} # set parameters from earlier runs
    mpc_params = {}
    include_runs = {"rl_agg"}
    r = Reformat(agg_params=agg_params, include_runs=include_runs)

    r.rl2baseline()
    r.rl_thetas()
    if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated
        r.plot_single_home2("Crystal-RXXFA") # pv_battery
        # r.plot_single_home2(type="base")
