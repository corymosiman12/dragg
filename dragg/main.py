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

    agg_params = {"alpha": [0.001, 0.005], "beta":[0.2], "epsilon":[0.1, 0.3], "rl_horizon":[], "mpc_disutility":[]} # set parameters from earlier runs
    mpc_params = {"mpc_hourly_steps": [4], "mpc_prediction_horizons": [1], "mpc_discomfort":[3.0, 10.0]}
    date_ranges = {"end_datetime": "2015-01-08 00"}
    # date_ranges = {}
    include_runs = {"baseline", "rl_agg"}
    add_outputs = {}
    # add_outputs = {}
    r = Reformat(add_outputs=add_outputs, agg_params=agg_params, mpc_params=mpc_params, include_runs=include_runs, date_ranges=date_ranges)
    # for i in r.parametrics:
    #     print(i['results'])
    r.rl2baseline()
    r.rl2baseline_error()
    r.rl_thetas()
    r.rl_qvals()
    r.all_rps()
    r.plot_single_home2("Crystal-RXXFA") # pv_battery
    # if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated

        # r.plo1t_single_home2(type="base")

        # r.plot_all_homes()
