# dragg
Distributed Resource AGGregation (DRAGG) implements centralized MPC for residential buildings using an aggregator and residential building owner (RBO) model.

# Setup
You will need to have docker and docker-compose installed.
1. Download data from the [NSRDB](https://maps.nrel.gov/nsrdb-viewer) for the location / year of your choice.  Only variable which needs to be selected is the `GHI`.  Make sure to also select `Half Hour Intervals`, as simulations start on the hour. Copy file and rename: `data/nsrdb.csv`, or change the default file name in `.env`
1. Copy the `data/config-template.json` to a new file: `config.json`
1. Change the parameters in the config file:
    - `total_number_homes` - int, total number of homes in study
    - `homes_battery` - int, number of homes with battery only
    - `homes_pv` - int, number of homes with pv only
    - `homes_pv_battery` - int, number of homes with pv and battery
    - `home_hvac_r_dist_low` - float, inclusive lower bound for home resistance, K/W
    - `home_hvac_r_dist_high` - float, exclusive upper bound for home resistance, K/W
    - `home_hvac_c_dist_low` - float, inclusive lower bound for home capacitance, J/K
    - `home_hvac_c_dist_high` - float, exclusive upper bound for home capacitance, J/K
    - `home_hvac_p_dist_low` - float, inclusive lower bound for hvac power, kW
    - `home_hvac_p_dist_high` - float, exclusive upper bound for hvac power, kW
    - `battery_max_rate` - float, maximum rate of charge / discharge for battery, kW
    - `battery_capacity` - float, energy capacity of battery, kW
    - `battery_lower_bound` - float, minimum proportion of the battery capacity to which to discharge, proportion btw. [0,1]
    - `battery_upper_bound` - float, minimum proportion of the battery capacity to which to discharge, proportion btw. [0,1]
    - `battery_charge_eff` - float, battery charging efficiency, proportion btw. [0,1]
    - `battery_discharge_eff` - float, battery discharging efficiency, proportion btw. [0,1]
    - `pv_area` - float, area of pv array, m2
    - `pv_efficiency` - float, pv efficiency, proportion btw. [0,1]
    - `start_datetime` - str, "%Y-%m-%d %H" format for when to start experiment
    - `end_datetime` - str, "%Y-%m-%d %H" format for when to end experiment
    - `prediction_horizons` - list of integers, the prediction horizons (hours) over which to test the experiment
1. Add a `.dragg` directory to your home directory: `mkdir ~/.dragg`

# References
1. http://dx.doi.org/10.1016/j.apenergy.2017.08.166
1. https://ieeexplore.ieee.org/abstract/document/8906600
