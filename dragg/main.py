from dragg.aggregator import Aggregator
from dragg.reformat import Reformat
from dragg.logger import Logger
from datetime import datetime
import json
import os
import sys
import toml

if __name__ == "__main__":
    data_dir = 'data'
    config_file = os.path.join(data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
    with open(config_file) as f:
        config = toml.load(f)

    a = Aggregator("single_q")
    a.run()

    agg_params = {"alpha": [], "beta":[1.0], "epsilon":[0.005], "rl_horizon":[], "mpc_disutility":[]} # set parameters from earlier runs
    mpc_params = {"mpc_hourly_steps": [4], "mpc_prediction_horizons": [1], "mpc_discomfort":[]}
    # date_ranges = {"end_datetime": "2015-01-08 00"}
    date_ranges = {}
    include_runs = {"rl_agg", "baseline"}
    add_outputs = {"single_q", "duel_q"}
    # add_outputs = {}
    r = Reformat(add_outputs=add_outputs, agg_params=agg_params, mpc_params=mpc_params, include_runs=include_runs, date_ranges=date_ranges)
    # for i in r.baselines:
    #     print(i['results'])
    # for i in r.parametrics:
    #     print(i['results'])
    # if config['simulation']['run_rl_agg']:
    # r.rl2baseline()
    # r.rl2baseline_error()
    # # r.rl_thetas()
    # r.rl_qvals()
    # r.all_rps()
    # r.plot_single_home2("Crystal-RXXFA") # pv_battery
    # r.plot_mu()
        # r.show_all()

    if config['simulation']['run_rl_simplified']:
        for i in r.parametrics:
            print(i['results'])
        r.rl_simplified()
        r.rl_thetas()
        r.rl_qvals()
        # r.rl_thetas()
    # if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated

        # r.plot_single_home2(type="base")

        # r.plot_all_homes()
