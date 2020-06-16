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

    r = Reformat()
    r.agg_params["alpha"] |= set([0.9])
    r.agg_params["beta"] |= set([0.49, 0.51, 0.52])
    r._setup_agg_params()
    r._setup_mpc_params()
    r._set_base_files()
    r._set_rl_files()

    # r.plot_single_home2("Ruth-1HV86") # base
    # r.plot_single_home2("Crystal-RXXFA") # pv_battery
    r.plot_single_home2(type="base")
    # r.plot_single_home2("Bruno-PVRNB") # pv_only
    # r.plot_single_home2("Jason-INS3S") # battery_only

    # r.rl_qtables(rl_q_file)
