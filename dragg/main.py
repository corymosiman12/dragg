from dragg.aggregator import Aggregator
from dragg.reformat import *
import json
import os

if __name__ == "__main__":
    a = Aggregator()
    a.run()

    config_file = os.path.join("data", os.environ.get('CONFIG_FILE', 'config.json'))
    with open(config_file, 'r') as f:
        config = json.load(f)
    start_dt = datetime.strptime(config["start_datetime"], '%Y-%m-%d %H')
    end_dt = datetime.strptime(config["end_datetime"], '%Y-%m-%d %H')
    date_folder = f"{start_dt.strftime('%Y-%m-%dT%H')}_{end_dt.strftime('%Y-%m-%dT%H')}"


    nHouses = config["total_number_homes"]
    mpcHorizon = config["agg_mpc_horizon"]
    mpc_folder = f"all-homes_{nHouses}-horizon_{mpcHorizon}"

    alphas = config["agg_learning_rate"]
    epsilons = config["agg_exploration_rate"]
    betas = config["rl_agg_discount_factor"]

    rlHorizon = config["rl_agg_time_horizon"]

    rl_file = os.path.join("outputs", date_folder, mpc_folder, "rl_agg", f"agg_horizon_{rlHorizon}-alpha_{alphas[0]}-epsilon_{epsilons[0]}-beta_{betas[0]}-results.json") # file used to plot house response
    rl_q_file = os.path.join("outputs", date_folder, mpc_folder, "rl_agg", f"agg_horizon_{rlHorizon}-alpha_{alphas[0]}-epsilon_{epsilons[0]}-beta_{betas[0]}-iter-results.json")

    r = Reformat([rl_file])

    for alpha in alphas:
        for epsilon in epsilons:
            for beta in betas:
                file = os.path.join("outputs", date_folder, mpc_folder, "rl_agg", f"agg_horizon_{rlHorizon}-alpha_{alpha}-epsilon_{epsilon}-beta_{beta}-results.json")
                name = f"alpha={alpha}, beta={beta}, epsilon={epsilon}"
                r.add_parametric(file, name)

    base_file = os.path.join("outputs", date_folder, mpc_folder, "baseline", "baseline-results.json")
    if os.path.isfile(base_file):
        r.add_baseline(os.path.join("outputs", date_folder, mpc_folder, "baseline", "baseline-results.json"), "baseline")

    r.rl2baseline(rl_file, rl_q_file)

    r.plot_single_home2("Ruth-1HV86") # base
    r.plot_single_home2("Crystal-RXXFA") # pv_battery
    r.plot_single_home2("Bruno-PVRNB") # pv_only
    r.plot_single_home2("Jason-INS3S") # battery_only
