# dragg
Distributed Resource AGGregation (DRAGG) implements centralized MPC for residential buildings using an aggregator and residential building owner (RBO) model.

# Setup
This can be run in two ways - using a local redis server or by deploying through Docker.

## General
1. Download data from the [NSRDB](https://maps.nrel.gov/nsrdb-viewer) for the location / year of your choice.  Only variable which needs to be selected is the `GHI`.  Select `Half Hour Intervals` for most accurate simulation, as simulations start on the hour. (Note that DRAGG will repeat environmental data for sub-30 minute intervals. For example, for 15 minute intervals the temperature at 0:00 is the same as the temperature at 0:15, and the temperature at 0:30 is the same as the temperature at 0:45.) Copy file and rename: `data/nsrdb.csv`, or change the default file name in `.env`
1. Copy the `data/config-template.toml` to a new file: `config.toml`

## Modify Config File
1. Change the parameters in the config file:
    * community
        - `total_number_homes` - int, total number of homes in study
        - `homes_battery` - int, number of homes with battery only
        - `homes_pv` - int, number of homes with pv only
        - `homes_pv_battery` - int, number of homes with pv and battery

    * home
        * home.hvac
            - `r_dist` - list, [lower, upper] bound for home resistance, kWh/K
            - `c_dist` - list, [lower, upper] bound for home capacitance, K/kW
            - `p_cool_dist` - list, [lower, upper] bound for hvac power, kW
            - `p_heat_dist` - list, [lower, upper] bound for hvac power, kW

        * home.wh
            - `r_dist` - list, [lower, upper] bound for water heater resistance, kWh/K
            - `c_dist` - list, [lower, upper] bound for water heater capacitance, K/kW
            - `p_dist` - list, [lower, upper] bound for water heater power, kW

        * home.battery
            - `max_rate` - float, maximum rate of charge / discharge for battery, kW
            - `capacity` - float, energy capacity of battery, kWh
            - `cap_bounds` - list, [lower, upper] proportional bounds on battery capacity, proportion btw. [0,1]
            - `charge_eff` - float, battery charging efficiency, proportion btw. [0,1]
            - `discharge_eff` - float, battery discharging efficiency, proportion btw. [0,1]

        * home.pv
            - `pv_area` - float, area of pv array, m2
            - `pv_efficiency` - float, pv efficiency, proportion btw. [0,1]

        * home.hems
            - `prediction_horizons` - list of hours for MPC prediction horizon, 0 = no MPC
            - `discomfort` - depricated
            - `disutility` - depricated
            - `price_uncertainty` - float

    * simulation
        - `start_datetime` - str, "%Y-%m-%d %H" format for when to start experiment
        - `end_datetime` - str, "%Y-%m-%d %H" format for when to end experiment
        - `random_seed` - int, set the seed variable for the experiment
        - `load_zone` - str, this corresponds to the ERCOT load zone from which to pull the TOU pricing info from
        - `check_type` - str, choice of 'pv_only', 'base', 'battery_only', 'pv_battery', 'all'. defines which homes to run, all will run all homes (typical)
        - `run_rbo_mpc` - bool, runs homes using MPC Home Energy Management Systems (HEMS), no reward price signal
        - `run_rl_agg` - bool, runs homes using MPC HEMS, uses RL designed reward price signal
        - `run_rl_simplified` - bool, runs homes against the rl_simplified

    * rl
        * rl.parameters
            - `learning_rate` - float, controls update rate of the policy and critic network
            - `discount_factor` - float, depreciation rate of future expected rewards
            - `batch_size` - int, number of replay episodes
            - `exploration_rate` - float, standard deviation of selected action from mu (best action according to policy)
            - `twin_q` - bool, whether or not to run two competing critic ("Q") networks

        * rl.utility
            - `rl_agg_action_horizon` - list, number of hours in advace to forecast reward price signal
            - `rl_agg_forecast_horizon` - int, number of timestep iterations to forecast the home energy use
            - (OPTION A) `base_price` - float, price for electricity
            - (OPTION B) `shoulder_times` - list: int (len=2), electric utility/aggregator time of use times for "shoulder price" tier (time of day - 24hr clock)
            - (OPTION B) `peak_times` - list: int (len=2), electric utility/aggregator time of use times for "peak price" tier (time of day - 24hr clock)
            - (OPTION B) `offpeak_price` - float, electric utility/aggregator time of use price for "offpeak price" tier ($/kWh)
            - (OPTION B) `shoulder_price` - float, electric utility/aggregator time of use price for "shoulder price" tier ($/kWh)
            - (OPTION B) `peak_price` - float, electric utility/aggregator time of use price for "peak price" tier ($/kWh)
            - `action_space` - list, max/min action taken by RL agent in designing price signal
            - `action_scale` - float, scale of reward price signal to actionspace (e.g. for actionspace = [-5, 5] and reward_price = [-0.05, 0.05] action_scale = 100)
            - `hourly_steps` - number of price signals per hour

        * rl.simplified
            - `response_rate` - float, determines the response rate of the simplified (linear) response model's response to the RL price signal
            - `offset` - not implemented

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
- Advised to use the caffiene package to keep the Python process running. (Otherwise Python pauses when Mac goes idle.)
  1. `$ homebrew cask install caffeine`
  1. Run `main.py` using the caffeinate command `$ caffeinate -i python main.py`
  1. The `-s` argument will keep Python running even when the Mac is asleep (lid closed) `$ caffeinate -s python main.py`

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
