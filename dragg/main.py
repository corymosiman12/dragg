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

    r = Reformat()
    r.agg_params["alpha"] |= set([0.1]) # add additional params from previous runs
    # r.agg_params["beta"] |= set([0.54, 0.53])
    # r.agg_params["rl_horizon"] |= set([1])

    r.set_mpc_folders() # sets folders with additional mpc_params specified above
    # r.set_base_files() # adds baseline files to list of things to plot
    # r.set_rl_files() # adds rl_agg with actual community response to list of things to plot
    r.set_simplified_files() # adds the rl_agg with a *SIMPLIFIED* community response to list of things to plot

    r.rl2baseline()
    r.rl_thetas()
    if r.config["run_rl_agg"] or r.config["run_agg_mpc"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated
        # r.plot_single_home2("Crystal-RXXFA") # pv_battery
        r.plot_single_home2(type="base")
