import os
import sys
import threading
from queue import Queue

import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
import toml
import random
import names
import string
import cvxpy as cp
import dccp
import itertools as it
import redis

# Local
from dragg.mpc_calc import MPCCalc
from dragg.redis_client import RedisClient
from dragg.logger import Logger
from dragg.my_agents import HorizonAgent

class Aggregator:
    def __init__(self, run_name=None):
        self.agg_log = Logger("aggregator")
        self.mpc_log = Logger("mpc_calc")
        self.forecast_log = Logger("forecaster")
        self.rlagent_log = Logger("rl_agent")
        self.data_dir = 'data'
        self.outputs_dir = os.path.join('outputs')
        if run_name:
            self.outputs_dir = os.path.join(self.outputs_dir, run_name)
        if not os.path.isdir(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
        self.ts_data_file = os.path.join(self.data_dir, os.environ.get('SOLAR_TEMPERATURE_DATA_FILE', 'nsrdb.csv'))
        self.spp_data_file = os.path.join(self.data_dir, os.environ.get('SPP_DATA_FILE', 'tou_data.xlsx'))
        self.required_keys = {
            "community": {"total_number_homes": None},
            "home": {
                "hvac": {
                    "r_dist": None,
                    "c_dist": None,
                    "p_cool_dist": None,
                    "p_heat_dist": None,
                    "temp_sp_dist": None,
                    "temp_deadband_dist": None
                },
                "wh": {
                    "r_dist": None,
                    "c_dist": None,
                    "p_dist": None,
                    "sp_dist": None,
                    "deadband_dist": None,
                    "size_dist": None,
                    "waterdraws": {
                        "n_big_draw_dist",
                        "n_small_draw_dist",
                        "big_draw_size_dist",
                        "small_draw_size_dist"
                    }
                },
                "battery": {
                    "max_rate": None,
                    "capacity": None,
                    "cap_bounds": None,
                    "charge_eff": None,
                    "discharge_eff": None,
                    "cons_penalty": None
                },
                "pv": {
                    "area": None,
                    "efficiency": None
                },
                "hems": {
                    "prediction_horizon": None,
                    "discomfort": None,
                    "disutility": None
                }
            },
            "simulation": {
                "start_datetime": None,
                "end_datetime": None,
                "random_seed": None,
                "load_zone": None,
                "check_type": None,
                "run_rbo_mpc": None,
                "run_rl_agg": None,
                "run_rl_simplified": None
            }
        }
        self.timestep = None  # Set by redis_set_initial_values
        self.iteration = None  # Set by redis_set_initial_values
        self.reward_price = None  # Set by redis_set_initial_values
        self.start_hour_index = None  # Set by calc_star_hour_index
        self.agg_load = 0 # Reset after each iteration
        self.baseline_data = {}
        self.baseline_agg_load_list = []  # Aggregate load at every timestep from the baseline run
        self.max_agg_load = None  # Set after baseline run, the maximum aggregate load over all the timesteps
        self.max_agg_load_list = []

        self.num_threads = 1
        self.start_dt = None  # Set by _set_dt
        self.end_dt = None  # Set by _set_dt
        self.hours = None  # Set by _set_dt
        self.dt = None  # Set by _set_dt
        self.num_timesteps = None  # Set by _set_dt
        self.all_homes = None  # Set by get_homes
        self.queue = Queue()
        self.redis_client = RedisClient()
        self.config = self._import_config()
        # self.step_size_coeff = self.config["step_size_coeff"] # removed for RL aggregator
        self.check_type = self.config['simulation']['check_type']  # One of: 'pv_only', 'base', 'battery_only', 'pv_battery', 'all'

        self.ts_data = self._import_ts_data()  # Temp: degC, RH: %, Pressure: mbar, GHI: W/m2
        self.tou_data = self._import_tou_data()  # SPP: $/kWh
        self.all_data = self.join_data()
        self._set_dt()
        self._build_tou_price()
        self.all_data.drop("ts", axis=1)

        self.all_rps = np.zeros(self.hours * self.dt)
        self.all_sps = np.zeros(self.hours * self.dt)

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.agg_log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)
        with open(self.config_file, 'r') as f:
            data = toml.load(f)
            d_keys = set(data.keys())
            req_keys = set(self.required_keys.keys())
            if not req_keys.issubset(d_keys):
                missing_keys = req_keys - d_keys
                self.agg_log.logger.error(f"{missing_keys} must be configured in the config file.")
                sys.exit(1)
            else:
                for subsystem in self.required_keys.keys():
                    req_keys = set(self.required_keys[subsystem].keys())
                    given_keys = set(data[subsystem].keys())
                    if not req_keys.issubset(given_keys):
                        missing_keys = req_keys - given_keys
                        self.agg_logger.error(f"Parameters for {subsystem}: {missing_keys} must be specified in the config file.")
                        sys.exit(1)
            return data

    def _set_dt(self):
        """
        Convert the start and end datetimes specified in the config file into python datetime
        objects.  Calculate the number of hours for which the simulation will run.
        :return:
        """
        try:
            self.start_dt = datetime.strptime(self.config['simulation']['start_datetime'], '%Y-%m-%d %H')
            self.end_dt = datetime.strptime(self.config['simulation']['end_datetime'], '%Y-%m-%d %H')
        except ValueError as e:
            self.agg_log.logger.error(f"Error parsing datetimes: {e}")
            sys.exit(1)
        self.hours = self.end_dt - self.start_dt
        self.hours = int(self.hours.total_seconds() / 3600)

        self.num_timesteps = self.hours * self.dt
        self.mask = (self.all_data.index >= self.start_dt) & (self.all_data.index < self.end_dt)
        self.agg_log.logger.info(f"Start: {self.start_dt.isoformat()}; End: {self.end_dt.isoformat()}; Number of hours: {self.hours}")

    def _import_ts_data(self):
        """
        Import timeseries data from file downloaded from NREL NSRDB.  The function removes the top two
        lines.  Columns which must be present: ["Year", "Month", "Day", "Hour", "Minute", "Temperature", "GHI"]
        Renames 'Temperature' to 'OAT'
        :return: pandas.DataFrame, columns: ts, GHI, OAT
        """
        if not os.path.exists(self.ts_data_file):
            self.agg_log.logger.error(f"Timeseries data file does not exist: {self.ts_data_file}")
            sys.exit(1)

        df = pd.read_csv(self.ts_data_file, skiprows=2)
        # df = df[df["Minute"] == 0]
        self.dt = int(self.config['rl']['utility']['hourly_steps'])
        self.dt_interval = 60 // self.dt
        reps = [np.ceil(self.dt/2) if val==0 else np.floor(self.dt/2) for val in df.Minute]
        df = df.loc[np.repeat(df.index.values, reps)]
        interval_minutes = self.dt_interval * np.arange(self.dt)
        n_intervals = len(df.index) // self.dt
        x = np.tile(interval_minutes, n_intervals)
        df.Minute = x
        df = df.astype(str)
        df['ts'] = df[["Year", "Month", "Day", "Hour", "Minute"]].apply(lambda x: ' '.join(x), axis=1)
        df = df.rename(columns={"Temperature": "OAT"})
        df["ts"] = df["ts"].apply(lambda x: datetime.strptime(x, '%Y %m %d %H %M'))
        df = df.filter(["ts", "GHI", "OAT"])
        df[["GHI", "OAT"]] = df[["GHI", "OAT"]].astype(int)
        return df.reset_index(drop=True)

    def _import_tou_data(self):
        """
        Settlement Point Price (SPP) data as extracted from ERCOT historical DAM Load Zone and Hub Prices.
        url: http://www.ercot.com/mktinfo/prices.
        Only keeps SPP data, converts to $/kWh.
        Subtracts 1 hour from time to be inline with 23 hour day as required by pandas.
        :return: pandas.DataFrame, columns: ts, SPP
        """
        if not os.path.exists(self.spp_data_file):
            self.agg_log.logger.error(f"TOU data file does not exist: {self.spp_data_file}")
            sys.exit(1)
        df_all = pd.read_excel(self.spp_data_file, sheet_name=None)
        k1 = list(df_all.keys())[0]
        df = df_all[k1]
        for k, v in df_all.items():
            if k == k1:
                pass
            else:
                df = df.append(v, ignore_index=True)

        df = df[df["Settlement Point"] == self.config['simulation']['load_zone']]
        df["Hour Ending"] = df["Hour Ending"].str.replace(':00', '')
        df["Hour Ending"] = df["Hour Ending"].apply(pd.to_numeric)
        df["Hour Ending"] = df["Hour Ending"].apply(lambda x: x - 1)
        df["Hour Ending"] = df["Hour Ending"].astype(str)
        df['ts'] = df[["Delivery Date", "Hour Ending"]].apply(lambda x: ' '.join(x), axis=1)
        df = df.drop(columns=['Delivery Date', 'Hour Ending', 'Repeated Hour Flag', 'Settlement Point'])
        df = df.rename(columns={"Settlement Point Price": "SPP"})
        col_order = ["ts", "SPP"]
        df = df[col_order]
        df[["ts"]] = df.loc[:, "ts"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H'))
        df[["SPP"]] = df.loc[:, "SPP"].apply(lambda x: x / 1000)
        return df.reset_index(drop=True)

    def _build_tou_price(self):
        try:
            sd_times = self.config["shoulder_times"]
            pk_times = self.config["peak_times"]
            op_price = float(self.config["offpeak_price"])
            sd_price = float(self.config["shoulder_price"])
            pk_price = float(self.config["peak_price"])
            self.all_data['tou'] = self.all_data['ts'].apply(lambda x: pk_price if (x.hour <= pk_times[1] and x.hour >= pk_times[0]) else (sd_price if x.hour <= sd_times[1] and x.hour >= sd_times[0] else op_price))
        except:
            self.all_data['tou'] = float(self.config['rl']['utility']['base_price'])

    def join_data(self):
        """
        Join the TOU, GHI, temp data into a single dataframe
        :return: pandas.DataFrame
        """
        df = pd.merge(self.ts_data, self.tou_data, how='outer', on='ts')
        df = df.fillna(method='ffill')
        return df.set_index('ts', drop=False)

    def _check_home_configs(self):
        base_homes = [e for e in self.all_homes if e["type"] == "base"]
        pv_battery_homes = [e for e in self.all_homes if e["type"] == "pv_battery"]
        pv_only_homes = [e for e in self.all_homes if e["type"] == "pv_only"]
        battery_only_homes = [e for e in self.all_homes if e["type"] == "battery_only"]
        if not len(base_homes) == self.config['community']['total_number_homes'] - self.config['community']['homes_battery'] - self.config['community']['homes_pv'] - self.config['community']['homes_pv_battery']:
            self.agg_log.logger.error("Incorrect number of base homes.")
            sys.exit(1)
        elif not len(pv_battery_homes) == self.config['community']['homes_pv_battery']:
            self.agg_log.logger.error("Incorrect number of base pv_battery homes.")
            sys.exit(1)
        elif not len(pv_only_homes) == self.config['community']['homes_pv']:
            self.agg_log.logger.error("Incorrect number of base pv_only homes.")
            sys.exit(1)
        elif not len(battery_only_homes) == self.config['community']['homes_battery']:
            self.agg_log.logger.error("Incorrect number of base pv_only homes.")
            sys.exit(1)
        else:
            self.agg_log.logger.info("Homes looking ok!")

    def reset_seed(self, new_seed):
        """
        Reset value for seed.
        :param new_seed: int
        :return:
        """
        self.config['simulation']['random_seed'] = new_seed

    def get_homes(self):
        homes_file = os.path.join(self.outputs_dir, f"all_homes-{self.config['community']['total_number_homes']}-config.json")
        if not self.config['community']['overwrite_existing'] and os.path.isfile(homes_file):
            with open(homes_file) as f:
                self.all_homes = json.load(f)
        else:
            self.create_homes()
        self._check_home_configs()

    def create_homes(self):
        """
        Given parameter distributions and number of homes of each type, create a list
        of dictionaries of homes with the parameters set for each home.
        :return:
        """
        # Set seed before sampling.  Will ensure home name and parameters
        # are the same throughout different runs
        np.random.seed(self.config['simulation']['random_seed'])
        random.seed(self.config['simulation']['random_seed'])

        # Define home and HVAC parameters
        home_r_dist = np.random.uniform(
            self.config['home']['hvac']['r_dist'][0],
            self.config['home']['hvac']['r_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_c_dist = np.random.uniform(
            self.config['home']['hvac']['c_dist'][0],
            self.config['home']['hvac']['c_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_hvac_p_cool_dist = np.random.uniform(
            self.config['home']['hvac']['p_cool_dist'][0],
            self.config['home']['hvac']['p_cool_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_hvac_p_heat_dist = np.random.uniform(
            self.config['home']['hvac']['p_heat_dist'][0],
            self.config['home']['hvac']['p_heat_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_hvac_temp_in_sp_dist = np.random.uniform(
            self.config['home']['hvac']['temp_sp_dist'][0],
            self.config['home']['hvac']['temp_sp_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_hvac_temp_in_db_dist = np.random.uniform(
            self.config['home']['hvac']['temp_deadband_dist'][0],
            self.config['home']['hvac']['temp_deadband_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_hvac_temp_in_min_dist = home_hvac_temp_in_sp_dist - 0.5 * home_hvac_temp_in_db_dist
        home_hvac_temp_in_max_dist = home_hvac_temp_in_sp_dist + 0.5 * home_hvac_temp_in_db_dist
        home_hvac_temp_init = []
        for i in range(len(home_hvac_temp_in_min_dist)):
            home_hvac_temp_init.append(home_hvac_temp_in_min_dist[i] + np.random.uniform(0, home_hvac_temp_in_db_dist[i]))

        # Define water heater parameters
        wh_r_dist = np.random.uniform(
            self.config['home']['wh']['r_dist'][0],
            self.config['home']['wh']['r_dist'][1],
            self.config['community']['total_number_homes']
        )
        wh_c_dist = np.random.uniform(
            self.config['home']['wh']['c_dist'][0],
            self.config['home']['wh']['c_dist'][1],
            self.config['community']['total_number_homes']
        )
        wh_p_dist = np.random.uniform(
            self.config['home']['wh']['p_dist'][0],
            self.config['home']['wh']['p_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_wh_temp_sp_dist = np.random.uniform(
            self.config['home']['wh']['sp_dist'][0],
            self.config['home']['wh']['sp_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_wh_temp_db_dist = np.random.uniform(
            self.config['home']['wh']['deadband_dist'][0],
            self.config['home']['wh']['deadband_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_wh_temp_min_dist = home_wh_temp_sp_dist - 0.5 * home_wh_temp_db_dist
        home_wh_temp_max_dist = home_wh_temp_sp_dist + 0.5 * home_wh_temp_db_dist
        home_wh_temp_init = []
        for i in range(len(home_wh_temp_max_dist)):
            home_wh_temp_init.append(home_wh_temp_min_dist[i] + np.random.uniform(0, home_wh_temp_db_dist[i]))

        # define water heater draw events
        home_wh_size_dist = np.random.uniform(
            self.config['home']['wh']['size_dist'][0],
            self.config['home']['wh']['size_dist'][1],
            self.config['community']['total_number_homes']
        )
        home_wh_size_dist = (home_wh_size_dist + 10) // 20 * 20 # more even numbers

        ndays = self.num_timesteps // (24 * self.dt) + 1
        daily_timesteps = 24 * self.dt

        home_wh_all_draw_timing_dist = []
        home_wh_all_draw_size_dist = []
        for i in range(self.config['community']['total_number_homes']):
            n_daily_draws = np.random.randint(self.config['home']['wh']['waterdraws']['n_big_draw_dist'][0], self.config['home']['wh']['waterdraws']['n_big_draw_dist'][1]+1)
            typ_draw_times = np.random.randint(0, 24*self.dt, n_daily_draws)
            perturbations = np.array([])
            for d in range(ndays):
                perturbations = np.concatenate((perturbations, (np.random.randint(-1 * self.dt, self.dt, n_daily_draws) + (d * daily_timesteps))))
            big_draw_times = (np.tile(typ_draw_times, ndays) + perturbations)
            big_draw_sizes = (np.random.uniform(self.config['home']['wh']['waterdraws']['big_draw_size_dist'][0], self.config['home']['wh']['waterdraws']['big_draw_size_dist'][1], ndays * n_daily_draws))

            n_daily_draws = np.random.randint(self.config['home']['wh']['waterdraws']['n_small_draw_dist'][0], self.config['home']['wh']['waterdraws']['n_small_draw_dist'][1]+1)
            typ_draw_times = np.random.randint(0, 24*self.dt, n_daily_draws)
            perturbations = np.array([])
            for d in range(ndays):
                perturbations = np.concatenate((perturbations, (np.random.randint(-3 * self.dt, 3 * self.dt, n_daily_draws) + (d * daily_timesteps))))
            small_draw_times = (np.tile(typ_draw_times, ndays) + perturbations)
            small_draw_sizes = (np.random.uniform(self.config['home']['wh']['waterdraws']['small_draw_size_dist'][0], self.config['home']['wh']['waterdraws']['small_draw_size_dist'][1], ndays * n_daily_draws))

            all_draw_times = np.concatenate((big_draw_times, small_draw_times))
            all_draw_sizes = np.concatenate((big_draw_sizes, small_draw_sizes))
            ind = np.argsort(all_draw_times)
            all_draw_times = all_draw_times[ind].tolist()
            all_draw_sizes = all_draw_sizes[ind].tolist()

            home_wh_all_draw_timing_dist.append(all_draw_times)
            home_wh_all_draw_size_dist.append(all_draw_sizes)

        all_homes = []

        # PV values are constant
        pv = {
            "area": self.config['home']['pv']['area'],
            "eff": self.config['home']['pv']['efficiency']
        }

        # battery values also constant
        battery = {
            "max_rate": self.config['home']['battery']['max_rate'],
            "capacity": self.config['home']['battery']['capacity'],
            "capacity_lower": self.config['home']['battery']['cap_bounds'][0] * self.config['home']['battery']['capacity'],
            "capacity_upper": self.config['home']['battery']['cap_bounds'][1] * self.config['home']['battery']['capacity'],
            "ch_eff": self.config['home']['battery']['charge_eff'],
            "disch_eff": self.config['home']['battery']['discharge_eff'],
            "batt_cons": self.config['home']['battery']['cons_penalty'],
            "e_batt_init": np.random.uniform(self.config['home']['battery']['cap_bounds'][0] * self.config['home']['battery']['capacity'],
                                            self.config['home']['battery']['cap_bounds'][1] * self.config['home']['battery']['capacity'])
        }

        i = 0
        # Define pv and battery homes
        for _ in range(self.config['community']['homes_pv_battery']):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "pv_battery",
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i],
                    "temp_in_min": home_hvac_temp_in_min_dist[i],
                    "temp_in_max": home_hvac_temp_in_max_dist[i],
                    "temp_in_sp": home_hvac_temp_in_sp_dist[i],
                    "temp_in_init": home_hvac_temp_init[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i],
                    "temp_wh_min": home_wh_temp_min_dist[i],
                    "temp_wh_max": home_wh_temp_max_dist[i],
                    "temp_wh_sp": home_wh_temp_sp_dist[i],
                    "temp_wh_init": home_wh_temp_init[i],
                    "tank_size": home_wh_size_dist[i],
                    "draw_times": home_wh_all_draw_timing_dist[i],
                    "draw_sizes": home_wh_all_draw_size_dist[i]
                },
                "battery": battery,
                "pv": pv
            })
            i += 1

        # Define pv only homes
        for _ in range(self.config['community']['homes_pv']):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "pv_only",
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i],
                    "temp_in_min": home_hvac_temp_in_min_dist[i],
                    "temp_in_max": home_hvac_temp_in_max_dist[i],
                    "temp_in_sp": home_hvac_temp_in_sp_dist[i],
                    "temp_in_init": home_hvac_temp_init[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i],
                    "temp_wh_min": home_wh_temp_min_dist[i],
                    "temp_wh_max": home_wh_temp_max_dist[i],
                    "temp_wh_sp": home_wh_temp_sp_dist[i],
                    "temp_wh_init": home_wh_temp_init[i],
                    "tank_size": home_wh_size_dist[i],
                    "draw_times": home_wh_all_draw_timing_dist[i],
                    "draw_sizes": home_wh_all_draw_size_dist[i]
                },
                "pv": pv
            })
            i += 1

        # Define battery only homes
        for _ in range(self.config['community']['homes_battery']):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "battery_only",
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i],
                    "temp_in_min": home_hvac_temp_in_min_dist[i],
                    "temp_in_max": home_hvac_temp_in_max_dist[i],
                    "temp_in_sp": home_hvac_temp_in_sp_dist[i],
                    "temp_in_init": home_hvac_temp_init[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i],
                    "temp_wh_min": home_wh_temp_min_dist[i],
                    "temp_wh_max": home_wh_temp_max_dist[i],
                    "temp_wh_sp": home_wh_temp_sp_dist[i],
                    "temp_wh_init": home_wh_temp_init[i],
                    "tank_size": home_wh_size_dist[i],
                    "draw_times": home_wh_all_draw_timing_dist[i],
                    "draw_sizes": home_wh_all_draw_size_dist[i]
                },
                "battery": battery
            })
            i += 1

        base_homes = self.config['community']['total_number_homes'] - self.config['community']['homes_battery'] - self.config['community']['homes_pv'] - self.config['community']['homes_pv_battery']
        for _ in range(base_homes):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "base",
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i],
                    "temp_in_min": home_hvac_temp_in_min_dist[i],
                    "temp_in_max": home_hvac_temp_in_max_dist[i],
                    "temp_in_sp": home_hvac_temp_in_sp_dist[i],
                    "temp_in_init": home_hvac_temp_init[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i],
                    "temp_wh_min": home_wh_temp_min_dist[i],
                    "temp_wh_max": home_wh_temp_max_dist[i],
                    "temp_wh_sp": home_wh_temp_sp_dist[i],
                    "temp_wh_init": home_wh_temp_init[i],
                    "tank_size": home_wh_size_dist[i],
                    "draw_times": home_wh_all_draw_timing_dist[i],
                    "draw_sizes": home_wh_all_draw_size_dist[i]
                }
            })
            i += 1

        self.all_homes = all_homes
        self.write_home_configs()

    def reset_baseline_data(self):
        self.baseline_agg_load_list = []
        for home in self.all_homes:
            self.baseline_data[home["name"]] = {
                "type": home["type"],
                "temp_in_sp": home["hvac"]["temp_in_sp"],
                "temp_wh_sp": home["wh"]["temp_wh_sp"],
                "temp_in_opt": [home["hvac"]["temp_in_init"]],
                "temp_wh_opt": [home["wh"]["temp_wh_init"]],
                "p_grid_opt": [],
                "p_load_opt": [],
                "hvac_cool_on_opt": [],
                "hvac_heat_on_opt": [],
                "wh_heat_on_opt": [],
                "cost_opt": [],
            }
            if 'pv' in home["type"]:
                self.baseline_data[home["name"]]["p_pv_opt"] = []
                self.baseline_data[home["name"]]["u_pv_curt_opt"] = []
            if 'battery' in home["type"]:
                self.baseline_data[home["name"]]["e_batt_opt"] = [home["battery"]["e_batt_init"]]
                self.baseline_data[home["name"]]["p_batt_ch"] = []
                self.baseline_data[home["name"]]["p_batt_disch"] = []

    def check_all_data_indices(self):
        """
        Ensure enough data exists in all_data such that MPC calcs can be made throughout
        the requested start and end period.
        :return:
        """
        if not self.start_dt >= self.all_data.index[0]:
            self.agg_log.logger.error("The start datetime must exist in the data provided.")
            sys.exit(1)
        if not self.end_dt + timedelta(hours=max(self.config['home']['hems']['prediction_horizon'])) <= self.all_data.index[-1]:
            self.agg_log.logger.error("The end datetime + the largest prediction horizon must exist in the data provided.")
            sys.exit(1)

    def calc_start_hour_index(self):
        """
        Since all_data is posted as a list, where 0 corresponds to the first hour in
        the dataframe, the number of hours between the start_dt and the above mentioned
        hour needs to be calculated.
        :return:
        """
        start_hour_index = self.start_dt - self.all_data.index[0]
        self.start_hour_index = int(start_hour_index.total_seconds() / 3600)

    def redis_set_initial_values(self):
        """
        Set the initial timestep, iteration, reward price, and horizon to redis
        :return:
        """
        self.timestep = 0

        self.e_batt_init = self.config['home']['battery']['capacity'] * self.config['home']['battery']['cap_bounds'][0]
        self.redis_client.conn.hset("initial_values", "e_batt_init", self.e_batt_init)
        self.redis_client.conn.set("start_hour_index", self.start_hour_index)
        self.redis_client.conn.hset("current_values", "timestep", self.timestep)

        if self.case == "agg_mpc":
            self.iteration = 0
            self.redis_client.conn.hset("current_values", "iteration", self.iteration)
            self.redis_client.conn.hset("current_values", "reward_price", self.reward_price.tolist())

        if self.case == "rl_agg" or self.case == "simplified":
            self.reward_price = np.zeros(self.util['rl_agg_horizon'] * self.dt)
            for val in self.reward_price.tolist():
                self.redis_client.conn.rpush("reward_price", val)

    def redis_set_state_for_previous_timestep(self):
        """
        This is used for the AGG MPC implementation during back and forth iterations with the
        individual home in order to ensure, regardless of the iteration, the home always solves
        the problem using the previous states.  The previous optimal vals set by each home after
        converging need to be reset to reflect previous optimal state.
        :return:
        """
        for home, vals in self.baseline_data.items():
            for k, v in vals.items():
                if k == "temp_in_opt":
                    self.redis_client.conn.hset(home, k, v[-1])
                elif k == "temp_wh_opt":
                    self.redis_client.conn.hset(home, k, v[-1])
                if 'battery' in vals['type'] and k == "e_batt_opt":
                    self.redis_client.conn.hset(home, k, v[-1])

    def redis_add_all_data(self):
        """
        Values for the timeseries data are written to Redis as a list, where the
        column names: [GHI, OAT, SPP] are the redis keys.  Each list is as long
        as the data in self.all_data, which is 8760.
        :return:
        """
        for c in self.all_data.columns.to_list():
            data = self.all_data[c]
            for val in data.values.tolist():
                self.redis_client.conn.rpush(c, val)

    def redis_set_current_values(self):
        self.redis_client.conn.hset("current_values", "timestep", self.timestep)

        if self.case == "agg_mpc":
            self.redis_client.conn.hset("current_values", "iteration", self.iteration)
        elif self.case == "rl_agg" or self.case == "simplified":
            self.all_sps[self.timestep] = self.agg_setpoint
            self.all_rps[self.timestep] = self.reward_price[0]
            self.redis_client.conn.lpop("reward_price")
            self.redis_client.conn.rpush("reward_price", self.reward_price[-1])

    def rl_update_reward_price(self):
        """
        Updates the reward price signal vector sent to the demand response community.
        :return:
        """
        self.reward_price[:-1] = self.reward_price[1:]
        self.reward_price[-1] = self.action/100

    def _gen_forecast(self):
        """ Forecasts the anticipated energy consumption at each timestep through
        the MPCCalc class. Uses the predetermined reward price signal + a reward
        price of 0 for any unforecasted reward price signals.
        :return: list of type float
        """
        for home in self.all_homes:
             if self.check_type == "all" or home["type"] == self.check_type:
                 self.queue.put(home)

        forecast_horizon = self.config['rl']['utility']['rl_agg_forecast_horizon']
        if forecast_horizon < 1:
            forecast = self.agg_load * np.ones(self.mpc['horizon'])
        else:
            forecast = []
            for t in range(forecast_horizon):
                worker = MPCCalc(self.queue, self.mpc['horizon'], self.dt, self.mpc['discomfort'], self.mpc['disutility'], self.case, self.redis_client, self.forecast_log)
                forecast.append(worker.forecast(0)) # optionally give .forecast() method an expected value for the next RP
        return forecast # returns all forecast values in the horizon

    def _gen_setpoint(self, time):
        """ Generates the setpoint of the RL utility. Dynamically sized for the
        number of houses in the community.
        :return: float
        @kyri
        """
        sp = self.config['community']['total_number_homes']*2.5 # increased for homes with plug load
        return sp

    def check_baseline_vals(self):
        for home, vals in self.baseline_data.items():
            if self.check_type == 'all':
                homes_to_check = self.all_homes
            else:
                homes_to_check = [x for x in self.all_homes if x["type"] == self.check_type]
            if home in homes_to_check:
                for k, v2 in vals.items():
                    if k in ["temp_in_opt", "temp_wh_opt", "e_batt_opt"] and len(v2) != self.hours + 1:
                        self.agg_log.logger.error(f"Incorrect number of hours. {home}: {k} {len(v2)}")
                    elif len(v2) != self.hours:
                        self.agg_log.logger.error(f"Incorrect number of hours. {home}: {k} {len(v2)}")

    def run_iteration(self):
        worker = MPCCalc(self.queue, self.mpc['horizon'], self.dt, self.mpc['discomfort'], self.mpc['disutility'], self.case, self.redis_client, self.mpc_log)
        worker.run()

        # Block in Queue until all tasks are done
        self.queue.join()

        self.agg_log.logger.info(f"Workers complete for timestep {self.timestep} of {self.num_timesteps}.")
        self.agg_log.logger.info(f"Number of threads: {threading.active_count()}.")
        self.agg_log.logger.info(f"Length of queue: {self.queue.qsize()}.")

        self.timestep += 1

    def collect_data(self):
        agg_load = 0
        agg_cost = 0
        for home in self.all_homes:
            if self.check_type == 'all' or home["type"] == self.check_type:
                vals = self.redis_client.conn.hgetall(home["name"])
                for k, v in vals.items():
                    self.baseline_data[home["name"]][k].append(float(v))
                agg_load += float(vals["p_grid_opt"])
                agg_cost += float(vals["cost_opt"])
        self.agg_load = agg_load
        self.agg_cost = agg_cost
        self.baseline_agg_load_list.append(self.agg_load)

    def run_baseline(self):
        self.agg_log.logger.info(f"Performing baseline run for horizon: {self.mpc['horizon']}")
        self.start_time = datetime.now()
        for t in range(self.num_timesteps):
            for home in self.all_homes:
                if self.check_type == "all" or home["type"] == self.check_type:
                    self.queue.put(home)
            self.redis_set_current_values()
            self.run_iteration()
            self.collect_data()
            self.timestep += 1

        # Write
        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {self.mpc['horizon']}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")
        self.check_baseline_vals()

    def summarize_baseline(self):
        """
        Get the maximum of the aggregate demand
        :return:
        """
        self.max_agg_load = max(self.baseline_agg_load_list)
        self.max_agg_load_list.append(self.max_agg_load)

        self.agg_log.logger.info(f"Max load list: {self.max_agg_load_list}")
        self.baseline_data["Summary"] = {
            "case": self.case,
            "start_datetime": self.start_dt.strftime('%Y-%m-%d %H'),
            "end_datetime": self.end_dt.strftime('%Y-%m-%d %H'),
            "solve_time": self.t_diff.total_seconds(),
            "horizon": self.mpc['horizon'],
            "num_homes": self.config['community']['total_number_homes'],
            "p_max_aggregate": self.max_agg_load,
            "p_grid_aggregate": self.baseline_agg_load_list,
            "SPP": self.all_data.loc[self.mask, "SPP"].values.tolist(),
            "OAT": self.all_data.loc[self.mask, "OAT"].values.tolist(),
            "GHI": self.all_data.loc[self.mask, "GHI"].values.tolist(),
            "TOU": self.all_data.loc[self.mask, "tou"].values.tolist(),
            "RP": self.all_rps.tolist(),
            "p_grid_setpoint": self.all_sps.tolist()
        }

    def write_outputs(self):
        # Write values for baseline run to file

        date_output = os.path.join(self.outputs_dir, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}")
        if not os.path.isdir(date_output):
            os.makedirs(date_output)

        mpc_output = os.path.join(date_output, f"{self.check_type}-homes_{self.config['community']['total_number_homes']}-horizon_{self.mpc['horizon']}-interval_{self.dt_interval}")
        if not os.path.isdir(mpc_output):
            os.makedirs(mpc_output)

        agg_output = os.path.join(mpc_output, f"{self.case}")
        if not os.path.isdir(agg_output):
            os.makedirs(agg_output)

        if self.case == "baseline":
            file_name = f"{self.case}_discomf-{self.mpc['discomfort']}-results.json"

        elif self.case == "agg_mpc":
            file_name = "results.json"
            f2 = os.path.join(agg_output, f"{self.check_type}-homes_{self.config['community']['total_number_homes']}-horizon_{self.mpc['horizon']}-iter-results.json")
            with open(f2, 'w+') as f:
                json.dump(self.agg_mpc_data, f, indent=4)

        else: # self.case == "rl_agg" or self.case == "simplified"
            run_name = f"agg_horizon_{self.util['rl_agg_horizon']}-alpha_{self.rl_params['alpha']}-epsilon_{self.rl_params['epsilon']}-beta_{self.rl_params['beta']}_batch-{self.rl_params['batch_size']}_disutil-{self.mpc['disutility']}_discomf-{self.mpc['discomfort']}"
            run_dir = os.path.join(agg_output, run_name)
            if not os.path.isdir(run_dir):
                os.makedirs(run_dir)
            for agent in self.rl_agents:
                agent.write_rl_data(run_dir)
            file = os.path.join(agg_output, run_name, "results.json")

        with open(file, 'w+') as f:
            json.dump(self.baseline_data, f, indent=4)

    def write_home_configs(self):
        # Write all home configurations to file
        ah = os.path.join(self.outputs_dir, f"all_homes-{self.config['community']['total_number_homes']}-config.json")
        with open(ah, 'w+') as f:
            json.dump(self.all_homes, f, indent=4)

    def set_agg_mpc_initial_vals(self):
        temp = []
        for h in range(self.hours):
            temp.append({
                "timestep": h,
                "reward_price": [],
                "agg_cost": [],
                "agg_load": []
            })
        return temp

    def run_agg_mpc(self):
        self.agg_log.logger.info(f"Performing AGG MPC run for horizon: {self.mpc['horizon']}")
        self.start_time = datetime.now()
        self.agg_mpc_data = self.set_agg_mpc_initial_vals()
        for t in range(self.num_timesteps):
            self.converged = False
            while True:
                for home in self.all_homes:
                    if self.check_type == "all" or home["type"] == self.check_type:
                        self.queue.put(home)
                self.redis_set_current_values()
                self.run_iteration()
                self.check_agg_mpc_data()
                self.update_agg_mpc_data()
                if self.converged:
                    self.agg_log.logger.info(f"Converged for ts: {self.timestep} after iter: {self.iteration}")
                    break
                self.reward_price = self.update_reward_price()
                self.iteration += 1
                self.agg_log.logger.info(f"Not converged for ts: {self.timestep}; iter: {self.iteration}; rp: {self.reward_price:.20f}")
                # time.sleep(5)
                if hour > 0:
                    self.redis_set_state_for_previous_timestep()
            self.collect_data()
            self.iteration = 0
            self.reward_price = 0

        # Write
        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {self.mpc['horizon']}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")
        self.check_baseline_vals()

    def run_rl_agg(self):

        self.agg_log.logger.info(f"Performing RL AGG (agg. horizon: {self.util['horizon']}, learning rate: {self.rl_params['alpha']}, discount factor: {self.rl_params['beta']}, exploration rate: {self.rl_params['epsilon']}) with MPC HEMS for horizon: {self.mpc['horizon']}")
        self.start_time = datetime.now()

        self.actionspace = self.config['rl']['utility']['action_space']
        self.baseline_agg_load_list = [0]

        self.forecast_load = self._gen_forecast()
        self.prev_forecast_load = self.forecast_load
        self.forecast_setpoint = self._gen_setpoint(self.timestep)
        self.agg_load = self.forecast_load[0] # approximate load for initial timestep
        self.agg_setpoint = self._gen_setpoint(self.timestep)

        horizon_agent = HorizonAgent(self.rl_params, self.rlagent_log)
        # next_timestep_agent = NextTSAgent(self.rl_params, self.rlagent_log)
        self.rl_agents = [horizon_agent]

        for t in range(self.num_timesteps):
            self.agg_setpoint = self._gen_setpoint(self.timestep // self.dt)
            self.prev_forecast_load = self.forecast_load
            self.forecast_load = [self.agg_load] # forecast current load at next timestep
            self.forecast_setpoint = self._gen_setpoint(self.timestep + 1)

            self.redis_set_current_values() # broadcast rl price to community

            for home in self.all_homes: # uncomment these for the actual model response
                 if self.check_type == "all" or home["type"] == self.check_type:
                     self.queue.put(home)
            self.run_iteration() # community response to broadcasted price (done in a single iteration)
            self.collect_data()

            self.reward_price[:-1] = self.reward_price[1:]
            self.reward_price[-1] = horizon_agent.train(self) / self.config['rl']['utility']['action_scale']

        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {self.mpc['horizon']}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")

    def test_response(self):
        """ Tests the RL agent using a linear model of the community's response
        to changes in the reward price.
        @kyri
        """
        c = self.config['rl']['simplified']['response_rate']
        k = self.config['rl']['simplified']['offset']
        if self.timestep == 0:
            self.agg_load = self.agg_setpoint + 0.1*self.agg_setpoint
        # self.agg_load = self.agg_load - c * self.reward_price[0] * self.agg_load # can't go negative
        self.agg_load = self.agg_load - c * self.reward_price[-1] * (self.agg_setpoint)
        # self.agg_load = self.agg_load + 0.1 * self.agg_load * np.random.randn()
        # self.agg_load = np.clip(self.agg_load, 0, 2*self.agg_setpoint)
        # if self.reward_price[0] >= -0.02 and self.reward_price[0] <= 0.02:
        #     self.agg_load = self.agg_load + 0.5*(self.agg_setpoint - self.agg_load)
        #self.agg_load = max(200,self.agg_load) # can't go above 200
        self.agg_cost = self.agg_load * self.reward_price[0]
        self.agg_log.logger.info(f"Iteration {self.timestep} finished. Aggregate load {self.agg_load}")
        self.timestep += 1

    def run_rl_agg_simplified(self):
        self.mpc['discomfort'] = self.config['home']['hems']['discomfort'][0]
        self.agg_log.logger.info(f"Performing RL AGG (agg. horizon: {self.util['rl_agg_horizon']}, learning rate: {self.rl_params['alpha']}, discount factor: {self.rl_params['beta']}, exploration rate: {self.rl_params['epsilon']}) with simplified community model.")
        self.start_time = datetime.now()

        self.actionspace = self.config['rl']['utility']['action_space']
        self.baseline_agg_load_list = [0]

        self.forecast_setpoint = self._gen_setpoint(self.timestep)
        self.forecast_load = [self.forecast_setpoint]
        self.prev_forecast_load = self.forecast_load

        self.agg_load = self.forecast_load[0] # approximate load for initial timestep
        self.agg_setpoint = self._gen_setpoint(self.timestep)

        horizon_agent = HorizonAgent(self.rl_params, self.rlagent_log)
        # next_timestep_agent = NextTSAgent(self.rl_params, self.rlagent_log)
        self.rl_agents = [horizon_agent]

        for t in range(self.num_timesteps):
            self.agg_setpoint = self._gen_setpoint(self.timestep // self.dt)
            self.prev_forecast_load = self.forecast_load
            self.forecast_load = [self.agg_load] # forecast current load at next timestep
            self.forecast_setpoint = self._gen_setpoint(self.timestep + 1)

            self.redis_set_current_values() # broadcast rl price to community
            self.test_response()
            self.baseline_agg_load_list.append(self.agg_load)

            self.reward_price[:-1] = self.reward_price[1:]
            self.reward_price[-1] = horizon_agent.train(self) / self.config['rl']['utility']['action_scale']

        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")

    def flush_redis(self):
        self.redis_client.conn.flushall()
        self.agg_log.logger.info("Flushing Redis")
        time.sleep(1)
        self.check_all_data_indices()
        self.calc_start_hour_index()
        self.redis_add_all_data()

    def run(self):
        self.agg_log.logger.info("Made it to Aggregator Run")
        self.get_homes()

        mpc_parameters = {"horizon": [int(i) for i in self.config['home']['hems']['prediction_horizon']],
                          "discomfort": [float(i) for i in self.config['home']['hems']['discomfort']],
                          "disutility": [float(i) for i in self.config['home']['hems']['disutility']]
                          }
        keys, values = zip(*mpc_parameters.items())
        mpc_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

        if self.config['simulation']['run_rbo_mpc']:
            # Run baseline MPC with N hour horizon, no aggregator
            # Run baseline with 1 hour horizon for non-MPC HEMS
            self.case = "baseline" # no aggregator
            for i in mpc_permutations:
                self.flush_redis()
                self.redis_set_initial_values()
                self.reset_baseline_data()
                self.run_baseline()
                self.summarize_baseline()
                self.write_outputs()

        if self.config['simulation']['run_rl_agg']:
            self.case = "rl_agg"

            util_parameters = {"horizon": [int(i) for i in self.config['rl']['utility']['rl_agg_action_horizon']]}
            keys, values = zip(*util_parameters.items())
            util_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

            rl_parameters = {"alpha": [float(i) for i in self.config['rl']['parameters']['learning_rate']],
                                "beta": [float(i) for i in self.config['rl']['parameters']['discount_factor']],
                                "epsilon": [float(i) for i in self.config['rl']['parameters']['exploration_rate']],
                                "batch_size": [int(i) for i in self.config['rl']['parameters']['batch_size']],
                                "twin_q": [self.config['rl']['parameters']['twin_q']]
                                }
            keys, values = zip(*rl_parameters.items())
            rl_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

            for self.mpc in mpc_permutations:
                print(self.mpc)
                for self.util in util_permutations:
                    for self.rl_params in rl_permutations:
                        self.flush_redis()
                        self.redis_set_initial_values()
                        self.reset_baseline_data()
                        self.run_rl_agg()
                        self.summarize_baseline()
                        self.write_outputs()

        if self.config['simulation']['run_rl_simplified']:
            self.case = "simplified"

            util_parameters = {"rl_agg_horizon": [int(i) for i in self.config['rl']['utility']['rl_agg_action_horizon']]}
            keys, values = zip(*util_parameters.items())
            util_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

            rl_parameters = {"alpha": [float(i) for i in self.config['rl']['parameters']['learning_rate']],
                                "beta": [float(i) for i in self.config['rl']['parameters']['discount_factor']],
                                "epsilon": [float(i) for i in self.config['rl']['parameters']['exploration_rate']],
                                "batch_size": [int(i) for i in self.config['rl']['parameters']['batch_size']],
                                "twin_q": [self.config['rl']['parameters']['twin_q']]
                                }
            keys, values = zip(*rl_parameters.items())
            rl_permutations = [dict(zip(keys, v)) for v in it.product(*values)]

            for self.mpc in mpc_permutations:
                for self.util in util_permutations:
                    for self.rl_params in rl_permutations:
                        self.flush_redis()
                        self.redis_set_initial_values()
                        self.reset_baseline_data()
                        self.run_rl_agg_simplified()
                        self.summarize_baseline()
                        self.write_outputs()
