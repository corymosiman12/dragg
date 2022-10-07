from dragg.aggregator import Aggregator
from dragg.reformat import Reformat

if __name__ == "__main__":

    a = Aggregator()
    a.run()

    # agg_params = {"alpha": [0.0625], "beta":[1.0], "epsilon":[0.05, 0.025], "rl_horizon":[], "mpc_disutility":[]} # set parameters from earlier runs
    # mpc_params = {"mpc_hourly_steps": [4], "mpc_prediction_horizons": [1], "mpc_discomfort":[]}
    # # date_ranges = {"end_datetime": "2015-01-08 00"}
    # date_ranges = {}
    # include_runs = {}
    # add_outputs = {}

    # r = Reformat(mpc_params={"mpc_discomfort":[]})
    # r.main() # use main to plot a suite of graphs
    # r.save_images() # saves the images
    # r.rl2baseline() # specific plots available through named methods
