from dragg.aggregator import Aggregator
from dragg.reformat import *
import json
import os

if __name__ == "__main__":
    a = Aggregator()
    a.run()

    # plots using whatever datetime cycle is in the config file
    config_file = os.path.join("data", os.environ.get('CONFIG_FILE', 'config.json'))
    with open(config_file, 'r') as f:
        config = json.load(f)
    start_dt = datetime.strptime(config["start_datetime"], '%Y-%m-%d %H')
    end_dt = datetime.strptime(config["end_datetime"], '%Y-%m-%d %H')

    print(config["end_datetime"])
    r = Reformat([os.path.join("outputs", "baseline", f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}-baseline_all-homes_20-horizon_8-results.json")])

    rlfile = os.path.join(r.outputs_dir, "rl_agg", f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}-rl_agg_all-homes_20-horizon_8-results.json")
    rl_qfile = os.path.join(r.outputs_dir, "rl_agg", f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}-rl_agg_all-homes_20-horizon_8-iter-results.json")
    # adds the baseline if you've run the baseline, otherwise continues without plotting.
    mpc_noaggfile = os.path.join(r.outputs_dir, "baseline", f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}-baseline_all-homes_20-horizon_8-results.json")
    baselinefile = os.path.join(r.outputs_dir, "baseline", f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}-baseline_all-homes_20-horizon_1-results.json")

    r.rl2baseline(rlfile, rl_qfile, mpc_noaggfile, baselinefile)

    r.plot_single_home2("Ruth-1HV86") # base
    r.plot_single_home2("Crystal-RXXFA") # pv_battery
    r.plot_single_home2("Bruno-PVRNB") # pv_only
    r.plot_single_home2("Jason-INS3S") # battery_only
