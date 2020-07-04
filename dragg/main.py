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

    agg_params = {"alpha": [0.001], "beta":[], "epsilon":[], "rl_horizon":[], "mpc_disutility":[]} # set parameters from earlier runs
    mpc_params = {}
    date_ranges = {}
    include_runs = {"baseline", "rl_agg"}
    r = Reformat(agg_params=agg_params, include_runs=include_runs, date_ranges=date_ranges)

    r.rl2baseline()
    r.rl2baseline_error()
    # r.rl_thetas()
    # r.rl_qvals()
    r.plot_single_home2("Crystal-RXXFA") # pv_battery
    # if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated

        # r.plo1t_single_home2(type="base")

        # r.plot_all_homes()
