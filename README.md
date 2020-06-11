# dragg
Distributed Resource AGGregation (DRAGG) implements centralized MPC for residential buildings using an aggregator and residential building owner (RBO) model.

# Setup
This can be run in two ways - using a local redis server or by deploying through Docker.

## General
1. Download data from the [NSRDB](https://maps.nrel.gov/nsrdb-viewer) for the location / year of your choice.  Only variable which needs to be selected is the `GHI`.  Make sure to also select `Half Hour Intervals`, as simulations start on the hour. Copy file and rename: `data/nsrdb.csv`, or change the default file name in `.env`
1. Copy the `data/config-template.json` to a new file: `config.json`

## Modify Config File
1. Change the parameters in the config file:
    - `total_number_homes` - int, total number of homes in study
    - `homes_battery` - int, number of homes with battery only
    - `homes_pv` - int, number of homes with pv only
    - `homes_pv_battery` - int, number of homes with pv and battery
    - `home_hvac_r_dist` - list, [lower, upper] bound for home resistance, kWh/K
    - `home_hvac_c_dist` - list, [lower, upper] bound for home capacitance, K/kW
    - `home_hvac_p_dist` - list, [lower, upper] bound for hvac power, kW
    - `wh_r_dist` - list, [lower, upper] bound for water heater resistance, kWh/K
    - `wh_c_dist` - list, [lower, upper] bound for water heater capacitance, K/kW
    - `wh_p_dist` - list, [lower, upper] bound for water heater power, kW
    - `alpha_beta_dist` - deprecated
    - `battery_max_rate` - float, maximum rate of charge / discharge for battery, kW
    - `battery_capacity` - float, energy capacity of battery, kWh
    - `battery_cap_bounds` - list, [lower, upper] proportional bounds on battery capacity, proportion btw. [0,1]
    - `battery_charge_eff` - float, battery charging efficiency, proportion btw. [0,1]
    - `battery_discharge_eff` - float, battery discharging efficiency, proportion btw. [0,1]
    - `pv_area` - float, area of pv array, m2
    - `pv_efficiency` - float, pv efficiency, proportion btw. [0,1]
    - `start_datetime` - str, "%Y-%m-%d %H" format for when to start experiment
    - `end_datetime` - str, "%Y-%m-%d %H" format for when to end experiment
    - `prediction_horizons` - list of integers, the prediction horizons over which to test the experiment, hours
    - `random_seed` - int, set the seed variable for the experiment
    - `load_zone` - str, this corresponds to the ERCOT load zone from which to pull the TOU pricing info from
    - `step_size_coeff` - float, proportion to increase the marginal demand by for AGG <--> RBO iterations
    - `max_load_threshold` - list, threshold limits under which the AGG will try to maintain demand, kW
    - `check_type` - str, choice of 'pv_only', 'base', 'battery_only', 'pv_battery', 'all'. defines which homes to run, all will run all homes (typical)
    - `temp_in_init` - float, initial indoor air temperature, C
    - ` temp_wh_init` - float, initial water heater temperature, C
    - `temp_sp` - list, [lower, upper] bounds for home air temperature setpoints, C
    - `wh_sp` - list, [lower, upper] bounds for water heater temperature setpoints, C
    - `run_baseline` - bool, whether to run horizon = 1 scenario
    - `run_rbo_mpc` - bool, whether to run RBO MPC for all prediction horizons provided
    - `run_agg_mpc` - bool, whether to run AGG MPC, only for single horizon specified next
    - `run_rl_agg` - bool, whether to run aggregator using Reinforcement Learning, uses a variety of parameters
    - `agg_mpc_horizon` - int, the prediction horizon to use for AGG MPC, hours
    - `agg_learning_rate` - list: float [0,1], learning rate of reinforcement learning aggregator
    - `agg_exploration_rate` - list: float [0,1], percent of decisions made by aggregator to be exploritory
    - `rl_agg_discount_factor` - list: float [0,1], depreciation rate on future states of the system (compared to the current state)
    - `shoulder_times` - list: int (len=2), electric utility/aggregator time of use times for "shoulder price" tier (time of day - 24hr clock)
    - `peak_times` - list: int (len=2), electric utility/aggregator time of use times for "peak price" tier (time of day - 24hr clock)
    - `offpeak_price` - float, electric utility/aggregator time of use price for "offpeak price" tier ($/kWh)
    - `shoulder_price` - float, electric utility/aggregator time of use price for "shoulder price" tier ($/kWh)
    - `peak_price` - float, electric utility/aggregator time of use price for "peak price" tier ($/kWh)
    - `action_space` - list: float (len=2), min/max reward price for real time pricing of electric utility rates
    - `rl_agg_time_horizon` - list: int >= 2, number of hours ahead of current timestep to forecast reward price of aggregator
    - `batch_size` - list: int, batch size of experience replay for reinforcement learning aggregator


## Local Redis (Recommended)
1. Install and run a local Redis server.
1. Best to put this in some virtualenv and install requirements:
- `$ cd /wherever/dragg`
- `$ pip install -r requirements.txt`
1. cd into the root directory for this project and pip install this as a local package. Recommended to install as an editable package using the -e argument:
- `$ cd /wherever/dragg`
- `$ pip install -e .`

1. Run `main.py` from the lower dragg directory
- `$ cd /wherever/dragg/dragg`
- `$ python main.py`

## Docker Compose
You will need to have docker and docker-compose installed, but do not need Redis running.
1. Add a `.dragg` directory to your home directory: `mkdir ~/.dragg`
1. From the root dragg directory, run docker compose
- `$ cd /wherever/dragg`
- `$ docker-compose build`
- `$ docker-compose up`

# Known Limitations / TODOs
- Hope to make into a Dash / plotly webapp
- Not distributed architecture - runs in single process, however, still communicates across Redis.  Could implement threading, celery, etc. for scaling workers based on nubmer of tasks. The separate classes help for the aggregator and the MPC solver
- Separate the weather forecasting for the MPC solver so that houses can forecast weather in real time rather than reading a historical JSON
- Although a MongoDB is included in the compose setup, it is not utilized.

# Notable changes from upstream development
- Introduction of Reinforcement Learning Aggregator (in addition to negotiating aggregator)
- Change of MPC houses to a duty cycle run time decision variable
- Change of MPC houses to include a discomfort objective (and hard constraints on system parameters)

# References
1. http://dx.doi.org/10.1016/j.apenergy.2017.08.166
1. https://ieeexplore.ieee.org/abstract/document/8906600
1. [Draft](docs/Final%20Project-CDM-001-DRAFT.pdf)
