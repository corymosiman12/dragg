import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys

import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
import toml
import random
import names
import string
import redis
import pathos
from pathos.pools import ProcessPool

# Local
from dragg.mpc_calc import MPCCalc, manage_home
import dragg.redis_client as rc
from dragg.logger import Logger

REDIS_URL = "redis://localhost"

class Aggregator:
    """
    The aggregator combines the power consumption from all homes in the simulation, and manages
    the simulation of each home (MPCCalc) object in parallel.
    """
    def __init__(self, start=None, end=None, redis_url=REDIS_URL):
        """
        :parameter start: optional override of the start simulation datetime, string "YYYY-MM-DD HH"
        :parameter end: optional override of the end simulation datetime, string "YYYY-MM-DD HH"
        :parameter redis_url: optional override of the Redis host URL (must align with MPCCalc REDIS_URL)
        """
        self.log = Logger("aggregator")
        self.data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
        self.outputs_dir = os.path.join(os.getcwd(), 'outputs')
        if not os.path.isdir(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
        self.ts_data_file = os.path.join(self.data_dir, os.environ.get('SOLAR_TEMPERATURE_DATA_FILE', 'nsrdb.csv'))
        self.spp_data_file = os.path.join(self.data_dir, os.environ.get('SPP_DATA_FILE', 'spp_data.xlsx'))

        self.required_keys = {
            "community":    {"total_number_homes"},
            "home": {
                "hvac":     {"r_dist", "c_dist", "p_cool_dist", "p_heat_dist", "temp_sp_dist", "temp_deadband_dist"},
                "wh":       {"r_dist", "c_dist", "p_dist", "sp_dist", "deadband_dist", "size_dist", "waterdraw_file"},
                "battery":  {"max_rate", "capacity", "cap_bounds", "charge_eff", "discharge_eff", "cons_penalty"},
                "pv":       {"area", "efficiency"},
                "hems":     {"prediction_horizon", "discomfort", "disutility"}
            },
            "simulation":   {"start_datetime", "end_datetime", "random_seed", "load_zone", "check_type", "run_rbo_mpc"},
            "agg":          {"base_price", "subhourly_steps"}
        }
        
        self.str_start_dt = start 
        self.str_end_dt = end

        self.timestep = None  # Set by redis_set_initial_values
        self.iteration = None  # Set by redis_set_initial_values
        self.reward_price = None  # Set by redis_set_initial_values
        self.start_hour_index = None  # Set by calc_star_hour_index
        self.agg_load = 0 # Reset after each iteration
        self.collected_data = {}
        self.baseline_agg_load_list = []  # Aggregate load at every timestep from the baseline run
        self.max_agg_load = None  # Set after baseline run, the maximum aggregate load over all the timesteps
        self.max_agg_load_list = []

        self.start_dt = None  # Set by _set_dt
        self.end_dt = None  # Set by _set_dt
        self.hours = None  # Set by _set_dt
        self.dt = None  # Set by _set_dt
        self.num_timesteps = None  # Set by _set_dt
        self.all_homes = None  # Set by create_homes
        self.redis_url = redis_url
        self.redis_client = rc.connection(redis_url)
        self.config = self._import_config()
        self.waterdraws_file = os.path.join(self.data_dir, self.config['home']['wh']['waterdraw_file'])

        self.check_type = self.config['simulation']['check_type']  # One of: 'pv_only', 'base', 'battery_only', 'pv_battery', 'all'

        self.thermal_trend = None
        self.max_daily_temp = None
        self.max_daily_ghi = None
        self.min_daily_temp = None
        self.prev_load = None
        self.dt = min(2, int(self.config['agg']['subhourly_steps'])) # temporary reporting issue with dt == 1
        self._set_dt()
        self.ts_data = self._import_ts_data() # Temp: degC, RH: %, Pressure: mbar, GHI: W/m2
        
        self.spp_data = self._import_spp_data() # SPP: $/kWh
        self.tou_data = self._build_tou_price() # TOU: $/kWh
        self.all_data = self.join_data()

        self.all_rps = np.zeros(self.num_timesteps)
        self.all_sps = np.zeros(self.num_timesteps)

        self.case = "baseline"
        self.start_time = datetime.now()

        self.overwrite_output = False
        self.daily_peak = 0

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)
        with open(self.config_file, 'r') as f:
            data = toml.load(f)
            d_keys = set(data.keys())
            req_keys = set(self.required_keys.keys())
            if not req_keys.issubset(d_keys):
                missing_keys = req_keys - d_keys
                self.log.logger.error(f"{missing_keys} must be configured in the config file.")
                sys.exit(1)
            else:
                for subsystem in self.required_keys.keys():
                    req_keys = set(self.required_keys[subsystem])
                    given_keys = set(data[subsystem])
                    if not req_keys.issubset(given_keys):
                        missing_keys = req_keys - given_keys
                        self.log.logger.error(f"Parameters for {subsystem}: {missing_keys} must be specified in the config file.")
                        sys.exit(1)
        self.log.logger.info(f"Set the version write out to {data['simulation']['named_version']}")
        return data

    def _set_dt(self):
        """
        Convert the start and end datetimes specified in the config file into python datetime
        objects.  Calculate the number of hours for which the simulation will run.
        :return:
        """
        try: 
            if not self.str_start_dt:
                self.str_start_dt = self.config["simulation"]["start_datetime"]
            if not self.str_end_dt:
                self.str_end_dt = self.config["simulation"]["end_datetime"]
            self.start_dt = datetime.strptime(self.str_start_dt, '%Y-%m-%d %H')
            self.end_dt = datetime.strptime(self.str_end_dt, '%Y-%m-%d %H')
        except ValueError as e:
            self.log.logger.error(f"Error parsing datetimes: {e}")
            sys.exit(1)
        self.hours = self.end_dt - self.start_dt
        self.hours = int(self.hours.total_seconds() / 3600)

        self.num_timesteps = int(np.ceil(self.hours * self.dt))
        self.log.logger.info(f"Start: {self.start_dt.isoformat()}; End: {self.end_dt.isoformat()}; Number of hours: {self.hours}")

    def _import_ts_data(self):
        """
        Import timeseries data from file downloaded from NREL NSRDB.  The function removes the top two
        lines.  Columns which must be present: ["Year", "Month", "Day", "Hour", "Minute", "Temperature", "GHI"]
        Renames 'Temperature' to 'OAT'
        :return: pandas.DataFrame, columns: ts, GHI, OAT
        """
        if not os.path.exists(self.ts_data_file):
            self.log.logger.error(f"Timeseries data file does not exist: {self.ts_data_file}")
            sys.exit(1)

        df = pd.read_csv(self.ts_data_file, skiprows=2)

        
        self.dt_interval = 60 // self.dt

        # read in original data
        df["ts"] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
        df = df.rename(columns={"Temperature": "OAT"})
        df.set_index('ts', inplace=True)

        # create interpolated index
        df_interp = pd.DataFrame(index=pd.date_range(start=df.index.min(),end=df.index.max(),
                                                          freq=f'{self.dt_interval}T'))

        # create merged data and interpolate to fill missing points
        df = pd.concat([df, df_interp]).sort_index().interpolate('linear')
        df = df[~df.index.duplicated(keep='first')]

        # filter to only interpolated data (exclude non-control datetimes)
        df.filter(df_interp.index, axis=0)
        df.bfill(inplace=True)

        if self.dt == 1:
            print(type(df["Minute"].iloc[2]), df["Minute"].iloc[2])
            df["Minute"] = 0.0
            df["ts"] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
            df.set_index('ts', inplace=True)

        self.oat = df['OAT'].to_numpy()
        self.ghi = df['GHI'].to_numpy()

        df["WEEKDAY"] = df.index.weekday
        df["Hour"] = df.index.hour + (df.index.minute/60)

        self.start_index = df.index.get_loc(self.start_dt, method='nearest') # the index of the start datetime w/r/t the entire year
        self.end_index = df.index.get_loc(self.end_dt, method='nearest')
        idx = df.index[self.start_index]
        idx_h = df.index[self.start_index + 4 * self.dt]
        idx_end = df.index[self.end_index]
        self.thermal_trend = df.loc[idx_h, 'OAT'] - df.loc[idx, 'OAT']
        self.max_daily_temp = max(df.loc[df.index.date >= self.start_dt.date()]["OAT"])
        self.min_daily_temp = min(df.loc[df.index.date >= self.start_dt.date()]["OAT"])
        self.max_daily_ghi = max(df.loc[df.index.date >= self.start_dt.date()]["GHI"])
        
        return df

    def _import_spp_data(self):
        """
        Settlement Point Price (SPP) data as extracted from ERCOT historical DAM Load Zone and Hub Prices.
        url: http://www.ercot.com/mktinfo/prices.
        Only keeps SPP data, converts to $/kWh.
        Subtracts 1 hour from time to be inline with 23 hour day as required by pandas.
        :return: pandas.DataFrame, columns: ts, SPP
        """
        if not self.config['agg']['spp_enabled']:
            return

        if not os.path.exists(self.spp_data_file):
            self.log.logger.error(f"SPP data file does not exist: {self.spp_data_file}")
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
        df["ts"] = datetime.strptime(df['ts'], '%m/%d/%Y %H')
        df["SPP"] = df['SPP'] / 1000
        df = df.set_index('ts')
        return df

    def _build_tou_price(self):
        df = pd.DataFrame(index=pd.date_range(start=self.start_dt, periods=self.hours, freq='H'))
        df['tou'] = float(self.config['agg']['base_price'])
        if self.config['agg']['tou_enabled'] == True:
            sd_times = [int(i) for i in self.config['agg']['tou']['shoulder_times']]
            pk_times = [int(i) for i in self.config['agg']['tou']['peak_times']]
            sd_price = float(self.config['agg']['tou']['shoulder_price'])
            pk_price = float(self.config['agg']['tou']['peak_price'])
            df['tou'] = np.where(df.index.hour.isin(range(pk_times[0],pk_times[1])), pk_price, float(self.config['agg']['base_price']))
            df['tou'] = np.where(df.index.hour.isin(range(sd_times[0],sd_times[1])), sd_price, float(self.config['agg']['base_price']))
        return df


    def join_data(self):
        """
        Join the TOU, GHI, temp data into a single dataframe
        :return: pandas.DataFrame
        """
        if self.config['agg']['spp_enabled']:
            df = pd.merge(self.ts_data, self.spp_data, how='outer', left_index=True, right_index=True)
        else:
            df = pd.merge(self.ts_data, self.tou_data, how='outer', left_index=True, right_index=True)
        df = df.fillna(method='ffill')
        self.mask = (df.index >= self.start_dt) & (df.index < self.end_dt)
        return df

    def _check_home_configs(self):
        base_homes = [e for e in self.all_homes if e['type'] == "base"]
        pv_battery_homes = [e for e in self.all_homes if e['type'] == "pv_battery"]
        pv_only_homes = [e for e in self.all_homes if e['type'] == "pv_only"]
        battery_only_homes = [e for e in self.all_homes if e['type'] == "battery_only"]
        if not len(base_homes) == (self.config['community']['total_number_homes']
                                - self.config['community']['homes_battery']
                                - self.config['community']['homes_pv']
                                - self.config['community']['homes_pv_battery']):
            self.log.logger.error("Incorrect number of base homes.")
            sys.exit(1)
        elif not len(pv_battery_homes) == self.config['community']['homes_pv_battery']:
            self.log.logger.error("Incorrect number of base pv_battery homes.")
            sys.exit(1)
        elif not len(pv_only_homes) == self.config['community']['homes_pv']:
            self.log.logger.error("Incorrect number of base pv_only homes.")
            sys.exit(1)
        elif not len(battery_only_homes) == self.config['community']['homes_battery']:
            self.log.logger.error("Incorrect number of base pv_only homes.")
            sys.exit(1)
        else:
            self.log.logger.info("Homes looking ok!")

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
            self.create_mpc_home_obj()
        self._check_home_configs()
        self.write_home_configs()

    def get_home_names(self):
        return [f"{names.get_first_name()}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}" for _ in range(self.config['community']['total_number_homes'])]

    def get_hvac_params(self):
        hvac = {
                "r": np.random.uniform(
                    self.config['home']['hvac']['r_dist'][0],
                    self.config['home']['hvac']['r_dist'][1]),
                "c": np.random.uniform(
                    self.config['home']['hvac']['c_dist'][0],
                    self.config['home']['hvac']['c_dist'][1]),
                "w": np.random.uniform(
                    self.config['home']['hvac']['window_eq_dist'][0],
                    self.config['home']['hvac']['window_eq_dist'][1]),
                "hvac_seer": self.config["home"]["hvac"]["seer"],
                "hvac_hspf": self.config["home"]["hvac"]["hspf"],
                "p_c": np.random.uniform(
                    self.config['home']['hvac']['p_cool_dist'][0],
                    self.config['home']['hvac']['p_cool_dist'][1]),
                "p_h": np.random.uniform(
                    self.config['home']['hvac']['p_heat_dist'][0],
                    self.config['home']['hvac']['p_heat_dist'][1]),
                "temp_in_sp": np.random.uniform(
                    self.config['home']['hvac']['temp_sp_dist'][0],
                    self.config['home']['hvac']['temp_sp_dist'][1]),
                "temp_in_db": np.random.uniform(
                    self.config['home']['hvac']['temp_deadband_dist'][0],
                    self.config['home']['hvac']['temp_deadband_dist'][1]),
                "temp_setback_delta": np.random.uniform(
                    self.config['home']['hvac']['temp_setback_delta'][0],
                    self.config['home']['hvac']['temp_setback_delta'][1])
            }

        hvac.update({
                "temp_in_min": hvac["temp_in_sp"] - 0.5 * hvac["temp_in_db"],
                "temp_in_max": hvac["temp_in_sp"] + 0.5 * hvac["temp_in_db"],
                "temp_in_init": hvac["temp_in_sp"] + np.random.uniform(-0.5,0.5) * hvac["temp_in_db"]
            }) 
        return hvac 

    def get_wh_params(self):
        wh = {
                "r": np.random.uniform(
                    self.config['home']['wh']['r_dist'][0],
                    self.config['home']['wh']['r_dist'][1]),
                "p": np.random.uniform(
                    self.config['home']['wh']['p_dist'][0],
                    self.config['home']['wh']['p_dist'][1]),
                "temp_wh_sp": np.random.uniform(
                    self.config['home']['wh']['sp_dist'][0],
                    self.config['home']['wh']['sp_dist'][1]),
                "temp_wh_db": np.random.uniform(
                    self.config['home']['wh']['deadband_dist'][0],
                    self.config['home']['wh']['deadband_dist'][1]),
                "tank_size": np.random.uniform(
                    self.config['home']['wh']['size_dist'][0],
                    self.config['home']['wh']['size_dist'][1]),   
            }

        waterdraw_df = pd.read_csv(self.waterdraws_file, index_col=0)
        waterdraw_df.index = pd.to_datetime(waterdraw_df.index, format='%Y-%m-%d %H:%M:%S')
        sigma = 0.2
        waterdraw_df = waterdraw_df.applymap(lambda x: x * (1 + sigma * np.random.randn()))
        waterdraw_df = waterdraw_df.resample(f'{self.dt_interval}T').sum()
        this_house_waterdraws = waterdraw_df[list(waterdraw_df.sample(axis='columns'))[0]].values.tolist()
        this_house_waterdraws = np.clip(this_house_waterdraws, 0, wh["tank_size"]).tolist()

        wh.update({
                "temp_wh_min": wh["temp_wh_sp"] - 0.5 * wh["temp_wh_db"],
                "temp_wh_max": wh["temp_wh_sp"] + 0.5 * wh["temp_wh_db"],
                "temp_wh_init": wh["temp_wh_sp"] + np.random.uniform(-0.5,0.5) * wh["temp_wh_db"],
                "draw_sizes": this_house_waterdraws
            })
        return wh

    def get_battery_params(self):
        battery = {
            "max_rate": np.random.uniform(self.config['home']['battery']['max_rate'][0],
                                            self.config['home']['battery']['max_rate'][1]),
            "capacity": np.random.uniform(self.config['home']['battery']['capacity'][0],
                                            self.config['home']['battery']['capacity'][1]),
            "capacity_lower": np.random.uniform(self.config['home']['battery']['lower_bound'][0],
                                            self.config['home']['battery']['lower_bound'][1]),
            "capacity_upper": np.random.uniform(self.config['home']['battery']['upper_bound'][0],
                                            self.config['home']['battery']['upper_bound'][1]),
            "ch_eff": np.random.uniform(self.config['home']['battery']['charge_eff'][0],
                                            self.config['home']['battery']['charge_eff'][1]),
            "disch_eff": np.random.uniform(self.config['home']['battery']['discharge_eff'][0],
                                            self.config['home']['battery']['discharge_eff'][1]),
            "e_batt_init": np.random.uniform(self.config['home']['battery']['lower_bound'][1],
                                            self.config['home']['battery']['upper_bound'][0])
        }
        return battery

    def get_pv_params(self):
        pv = {
            "area": np.random.uniform(self.config['home']['pv']['area'][0],
                                    self.config['home']['pv']['area'][1]),
            "eff": np.random.uniform(self.config['home']['pv']['efficiency'][0],
                                    self.config['home']['pv']['efficiency'][1])
        }
        return pv

    def get_hems_params(self):
        responsive_hems = {
            "horizon": self.config['home']['hems']['prediction_horizon'],
            "hourly_agg_steps": self.dt,
            "sub_subhourly_steps": self.config['home']['hems']['sub_subhourly_steps'],
            "solver": self.config['home']['hems']['solver'],
            "discount_factor": self.config['home']['hems']['discount_factor'],
            "weekday_occ_schedule": self.config['home']['hems']['weekday_occ_schedule'], # depricated
            "schedule_group": random.choices(list(self.config['community']['schedules'].keys()), 
                weights=list(self.config['community']['schedules'].values()))[0]
        }
        offset = -3 if responsive_hems['schedule_group'] == 'early_birds' else 3 if responsive_hems['schedule_group'] == 'night_owls' else 0
        responsive_hems.update({
            "typ_leave": np.random.randint(8,10) + offset,
            "typ_return": np.random.randint(18,20) + offset 
            })
        return responsive_hems

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

        daily_timesteps = int(24 * self.dt)
        ndays = self.num_timesteps // daily_timesteps + 1

        self.all_homes = []

        if not os.path.isdir(os.path.join('home_logs')):
            os.makedirs('home_logs')

        all_names = self.get_home_names()

        i = 0
        # Define pv and battery homes
        num_pv_battery_homes = self.config['community']['homes_pv_battery']
        num_pv_homes = self.config['community']['homes_pv']
        num_battery_homes = self.config['community']['homes_battery']
        num_base_homes = self.config['community']['total_number_homes'] - num_battery_homes - num_pv_homes - num_pv_battery_homes

        

        for j in range(num_pv_homes):
            name = all_names.pop()
            self.all_homes += [{
                    "name": name,
                    "type": "pv_only",
                    "hvac": self.get_hvac_params(),
                    "wh": self.get_wh_params(),
                    "hems": self.get_hems_params(),
                    "pv": self.get_pv_params()
                }]

        for j in range(num_battery_homes):
            name = all_names.pop()
            self.all_homes += [{
                    "name": name,
                    "type": "battery_only",
                    "hvac": self.get_hvac_params(),
                    "wh": self.get_wh_params(),
                    "hems": self.get_hems_params(),
                    "battery": self.get_battery_params()
                }]

        for j in range(num_pv_battery_homes):
            name = all_names.pop()
            self.all_homes += [{
                    "name": name,
                    "type": "pv_battery",
                    "hvac": self.get_hvac_params(),
                    "wh": self.get_wh_params(),
                    "hems": self.get_hems_params(),
                    "battery": self.get_battery_params(),
                    "pv": self.get_pv_params()
                }]

        for j in range(num_base_homes):
            name = all_names.pop()
            self.all_homes += [{
                    "name": name,
                    "type": "base",
                    "hvac": self.get_hvac_params(),
                    "wh": self.get_wh_params(),
                    "hems": self.get_hems_params(),
                }]
            
        self.contribution2peak = {home['name']:0 for home in self.all_homes}

        self.all_homes_obj = []
        self.max_poss_load = 0
        self.min_poss_load = 0

    def create_mpc_home_obj(self):
        for home in self.all_homes:
            home_obj = MPCCalc(home, self.redis_url)
            self.all_homes_obj += [home_obj]
            self.max_poss_load += home_obj.max_load

    def reset_collected_data(self):
        self.timestep = 0
        self.baseline_agg_load_list = []
        for home in self.all_homes:
            self.collected_data[home["name"]] = {
                "type": home["type"],
                "temp_in_sp": home["hvac"]["temp_in_sp"],
                "temp_wh_sp": home["wh"]["temp_wh_sp"],
                "temp_in_opt": [home["hvac"]["temp_in_init"]],
                "temp_wh_opt": [home["wh"]["temp_wh_init"]],
                "p_grid_opt": [],
                "forecast_p_grid_opt": [],
                "p_load_opt": [],
                "hvac_cool_on_opt": [],
                "hvac_heat_on_opt": [],
                "wh_heat_on_opt": [],
                "cost_opt": [],
                "waterdraws": [],
                "correct_solve": [],
                "t_in_min":[],
                "t_in_max":[],
            }
            if 'pv' in home["type"]:
                self.collected_data[home["name"]]["p_pv_opt"] = []
                self.collected_data[home["name"]]["u_pv_curt_opt"] = []
            if 'battery' in home["type"]:
                self.collected_data[home["name"]]["e_batt_opt"] = [home["battery"]["e_batt_init"]]
                self.collected_data[home["name"]]["p_batt_ch"] = []
                self.collected_data[home["name"]]["p_batt_disch"] = []
            if True: # 'ev' in home["type"]:
                self.collected_data[home["name"]]["e_ev_opt"] = [16.0]
                self.collected_data[home["name"]]["p_ev_ch"] = []
                self.collected_data[home["name"]]["p_ev_disch"] = []
                self.collected_data[home["name"]]["p_v2g"] = []

    def check_all_data_indices(self):
        """
        Ensure enough data exists in all_data such that MPC calcs can be made throughout
        the requested start and end period.
        :return: None
        """
        if not self.start_dt >= self.all_data.index[0]:
            self.log.logger.error("The start datetime must exist in the data provided.")
            sys.exit(1)
        if not self.end_dt + timedelta(hours=self.config['home']['hems']['prediction_horizon']) <= self.all_data.index[-1]:
            self.log.logger.error("The end datetime + the largest prediction horizon must exist in the data provided.")
            sys.exit(1)

    def calc_start_hour_index(self):
        """
        Since all_data is posted as a list, where 0 corresponds to the first hour in
        the dataframe, the number of hours between the start_dt and the above mentioned
        hour needs to be calculated.
        :return: None
        """

        start_hour_index = self.start_dt - self.all_data.index[0]
        self.start_hour_index = int(start_hour_index.total_seconds() / 3600)

    def redis_set_initial_values(self):
        """
        Set the initial timestep, iteration, reward price, and horizon to redis
        :return: None
        """
        self.timestep = 0

        self.redis_client.set("start_hour_index", self.start_hour_index)
        self.redis_client.hset("current_values", "timestep", self.timestep)
        self.redis_client.hset("current_values", "start_index", self.start_index)
        self.redis_client.hset("current_values", "end_index", self.end_index)
        self.reward_price = np.zeros(self.config['agg']['rl']['action_horizon'] * self.dt)
        self.redis_client.rpush("reward_price", *self.reward_price.tolist())

    def redis_add_all_data(self):
        """
        Values for the timeseries data are written to Redis as a list, where the
        column names: [GHI, OAT, SPP] are the redis keys.  Each list is as long
        as the data in self.all_data, which is 8760 for default config file.
        :return: None
        """
        self.all_data.bfill(inplace=True)
        for c in self.all_data.columns.to_list():
            self.redis_client.delete(c)
            self.redis_client.rpush(c, *self.all_data[c].values.tolist())

    def redis_set_current_values(self):
        """
        Sets the current values of the utility agent (reward price).
        :return: None
        """
        self.redis_client.hset("current_values", "timestep", self.timestep)

        if 'rl' in self.case:
            self.all_sps[self.timestep] = self.agg_setpoint
            self.all_rps[self.timestep] = self.reward_price[0]
            self.redis_client.delete("reward_price")
            self.redis_client.rpush("reward_price", *self.reward_price)

    def gen_setpoint(self):
        """
        Generates the setpoint of the RL utility. Dynamically sized for the
        number of houses in the community.
        :return: float
        """
        if self.timestep < 2:
            self.tracked_loads = [0.5 * self.max_poss_load] * self.config['agg']['rl']['prev_timesteps']
            self.max_load = -float("inf")
            self.min_load = float("inf")
        else:
            self.tracked_loads[:-1] = self.tracked_loads[1:]
            self.tracked_loads[-1] = self.agg_load
        self.avg_load = np.average(self.tracked_loads)
        if self.agg_load > self.max_load or self.timestep % 24 == 0:
            self.max_load = self.agg_load
        if self.agg_load < self.min_load or self.timestep % 24 == 0:
            self.min_load = self.agg_load
        sp = self.avg_load
        return sp

    def check_baseline_vals(self):
        for home, vals in self.collected_data.items():
            if self.check_type == 'all':
                homes_to_check = self.all_homes
            else:
                homes_to_check = [x for x in self.all_homes if x["type"] == self.check_type]
            if home in homes_to_check:
                for k, v2 in vals.items():
                    if k in ["temp_in_opt", "temp_wh_opt", "e_batt_opt"] and len(v2) != self.hours + 1:
                        self.log.logger.error(f"Incorrect number of hours. {home}: {k} {len(v2)}")
                    elif len(v2) != self.hours:
                        self.log.logger.error(f"Incorrect number of hours. {home}: {k} {len(v2)}")

    def run_iteration(self):
        """
        Calls the MPCCalc class to calculate the control sequence and power demand
        from all homes in the community. Threaded, using pathos
        :return: None
        """
        self.thermal_trend = self.oat[self.timestep + 4] - self.oat[self.timestep]
        day_of_year = self.timestep // (self.dt * 24)
        self.max_daily_temp = max(self.oat[day_of_year*(self.dt*24):(day_of_year+1)*(self.dt*24)])
        self.min_daily_temp = min(self.oat[day_of_year*(self.dt*24):(day_of_year+1)*(self.dt*24)])
        self.max_daily_ghi = max(self.ghi[day_of_year*(self.dt*24):(day_of_year+1)*(self.dt*24)])

        pool = ProcessPool(nodes=self.config['simulation']['n_nodes']) # open a pool of nodes
        results = pool.map(manage_home, self.mpc_players)

        # self.timestep = (self.timestep + 1) % self.num_timesteps

    def collect_data(self):
        """
        Collects the data passed by the community redis connection.
        :return: None
        """
        agg_load = 0
        agg_cost = 0
        
        house_load = {}
        self.forecast_house_load = []
        
        # if self.timestep % (24 * self.dt) == 0: # at the end of the day
        #     self.daily_peak = 0.01 # remove if we want the max peak of the simulation
        #     for k,v in self.contribution2peak.items():
        #         self.redis_client.hset("peak_contribution", k, v)

        for home in self.all_homes:
            if self.check_type == 'all' or home["type"] == self.check_type:
                vals = self.redis_client.hgetall(home["name"])
                for k, v in vals.items():
                    opt_keys = ["p_grid_opt", "forecast_p_grid_opt", "p_load_opt", "temp_in_opt", "temp_wh_opt", "hvac_cool_on_opt", "hvac_heat_on_opt", "wh_heat_on_opt", "cost_opt", "waterdraws", "correct_solve", "t_in_max", "t_in_min"]
                    if 'pv' in home["type"]:
                        opt_keys += ['p_pv_opt','u_pv_curt_opt']
                    if 'battery' in home["type"]:
                        opt_keys += ['p_batt_ch', 'p_batt_disch', 'e_batt_opt']
                    if True: #'ev' in home["type"]:
                        opt_keys += ['p_ev_ch', 'p_ev_disch', 'p_v2g', 'e_ev_opt']
                    if k in opt_keys:
                        self.collected_data[home["name"]][k].append(float(v))
                house_load.update({home['name']:float(vals["p_grid_opt"])})
                self.forecast_house_load.append(float(vals["forecast_p_grid_opt"]))
                agg_cost += float(vals["cost_opt"])
        self.agg_load = sum(house_load.values())

        if self.agg_load >= self.daily_peak and self.agg_load >= 0.1:
            self.daily_peak = self.agg_load
            self.contribution2peak = {k:house_load[k]/self.daily_peak for k,v in self.contribution2peak.items()}
            for k,v in self.contribution2peak.items():
                self.redis_client.hset("peak_contribution", k, v)

        self.forecast_load = np.sum(self.forecast_house_load)
        self.agg_cost = agg_cost
        self.baseline_agg_load_list.append(self.agg_load)
        self.agg_setpoint = self.gen_setpoint()
        self.log.logger.info(f"At time t={self.timestep} aggregate load is {round(self.agg_load,2)} kW.")
        self.timestep = (self.timestep + 1) % self.num_timesteps

    def run_baseline(self):
        """
        Runs the baseline simulation comprised of community of HEMS controlled homes.
        Utilizes MPC parameters specified in config file.
        (For no MPC in HEMS specify the MPC prediction horizon as 0.)
        :return: None
        """
        self.log.logger.info(f"Performing baseline run for horizon: {self.config['home']['hems']['prediction_horizon']}")
        self.start_time = datetime.now()

        self.mpc_players = []
        for home in self.all_homes_obj:
            if self.check_type == "all" or home.type == self.check_type:
                self.mpc_players += [home]
        for t in range(self.num_timesteps):
            self.redis_set_current_values()
            self.run_iteration()
            self.collect_data()

            if (t+1) % (self.checkpoint_interval) == 0: # weekly checkpoint
                self.log.logger.info("Creating a checkpoint file.")
                self.write_outputs()

    def my_summary(self):
        return

    def summarize_baseline(self):
        """
        Get the maximum of the aggregate demand for each simulation.
        :return: None
        """
        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.log.logger.info(f"Horizon: {self.config['home']['hems']['prediction_horizon']}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")

        self.max_agg_load = max(self.baseline_agg_load_list)
        self.max_agg_load_list.append(self.max_agg_load)

        self.collected_data["Summary"] = {
            "case": self.case,
            "start_datetime": self.start_dt.strftime('%Y-%m-%d %H'),
            "end_datetime": self.end_dt.strftime('%Y-%m-%d %H'),
            "num_timesteps": self.num_timesteps,
            "solve_time": self.t_diff.total_seconds(),
            "horizon": self.config['home']['hems']['prediction_horizon'],
            "num_homes": self.config['community']['total_number_homes'],
            "p_max_aggregate": self.max_agg_load,
            "p_grid_aggregate": self.baseline_agg_load_list,
            "OAT": self.all_data.loc[self.mask, "OAT"].values.tolist(),
            "GHI": self.all_data.loc[self.mask, "GHI"].values.tolist(),
            "RP": self.all_rps.tolist(),
            "p_grid_setpoint": self.all_sps.tolist()
        }

        self.my_summary()

        if self.config['agg']['spp_enabled']:
            self.collected_data["Summary"]["SPP"] = self.all_data.loc[self.mask, "SPP"].values.tolist()
        else:
            self.collected_data["Summary"]["TOU"] = self.all_data.loc[self.mask, "tou"].values.tolist()

    def set_run_dir(self):
        """
        Sets the run directoy based on the start/end datetime, community and home configs,
        and the named version.
        :return: none
        """
        if not self.overwrite_output:
            date_output = os.path.join(self.outputs_dir, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}")
            mpc_output = os.path.join(date_output, f"{self.check_type}-homes_{self.config['community']['total_number_homes']}-horizon_{self.config['home']['hems']['prediction_horizon']}-interval_{self.dt_interval}-{self.dt_interval // self.config['home']['hems']['sub_subhourly_steps']}-solver_{self.config['home']['hems']['solver']}")

            self.run_dir = os.path.join(mpc_output, f"version-{self.version}")
            if not os.path.isdir(self.run_dir):
                os.makedirs(self.run_dir)
        else:
            self.run_dir = self.outputs_dir

    def write_outputs(self):
        """
        Writes values for simulation run to a json file for later reference. Is
        called at the end of the simulation run period and optionally at a checkpoint period.
        :return: None
        """
        self.summarize_baseline()

        if not self.overwrite_output:
            case_dir = os.path.join(self.run_dir, self.case)
            if not os.path.isdir(case_dir):
                os.makedirs(case_dir)
            file = os.path.join(case_dir, "results.json")
        else:
            file = os.path.join(self.outputs_dir, "results.json")
        with open(file, 'w+') as f:
            json.dump(self.collected_data, f, indent=4)

    def write_home_configs(self):
        """
        Writes all home configurations to file at the initialization of the
        simulation for later reference.
        :return: None
        """
        ah = os.path.join(self.outputs_dir, f"all_homes-{self.config['community']['total_number_homes']}-config.json")
        with open(ah, 'w+') as f:
            json.dump(self.all_homes + [{"start_dt":self.str_start_dt, "end_dt":self.str_end_dt, "num_timesteps":self.num_timesteps}], f, indent=4)

    def set_agg_mpc_initial_vals(self):
        """
        Creates a dictionary to store values at each timestep for non-RL runs.
        :return: Dictionary
        """
        temp = []
        for h in range(self.hours):
            temp.append({
                "timestep": h,
                "reward_price": [],
                "agg_cost": [],
                "agg_load": []
            })
        return temp

    def set_dummy_rl_parameters(self):
        self.tracked_loads = self.config['community']['house_p_avg']*self.config['community']['total_number_homes']*np.ones(12)

    def setup_rl_agg_run(self):
        self.flush_redis()

        self.mpc_players = []
        for home in self.all_homes_obj:
            if self.check_type == "all" or home["type"] == self.check_type:
                self.mpc_players += [home]

        # self.log.logger.info(f"Performing RL AGG (agg. horizon: {self.util['rl_agg_horizon']}, learning rate: {self.rl_params['alpha']}, discount factor: {self.rl_params['beta']}, exploration rate: {self.rl_params['epsilon']}) with MPC HEMS for horizon: {self.config['home']['hems']['prediction_horizon']}")
        self.start_time = datetime.now()

        self.baseline_agg_load_list = [0]
        self.all_rewards = []

        self.forecast_load = 3*len(self.all_homes_obj)
        self.prev_forecast_load = self.forecast_load
        self.forecast_setpoint = self.gen_setpoint()
        self.agg_load = self.forecast_load # approximate load for initial timestep
        self.agg_setpoint = self.gen_setpoint()

        self.redis_set_current_values()

    def flush_redis(self):
        """
        Cleans all information stored in the Redis server. (Including environmental
        and home data.)
        :return: None
        """
        self.redis_client.flushall()
        self.log.logger.info("Flushing Redis")
        time.sleep(1)
        self.check_all_data_indices()
        self.calc_start_hour_index()
        self.redis_add_all_data()
        self.redis_set_initial_values()

    def run(self):
        """
        Runs simulation(s) specified in the config file with all combinations of
        parameters specified in the config file.
        :return: None
        """
        self.log.logger.info("Made it to Aggregator Run")

        self.checkpoint_interval = 500 # default to checkpoints every 1000 timesteps
        if self.config['simulation']['checkpoint_interval'] == 'hourly':
            self.checkpoint_interval = self.dt
        elif self.config['simulation']['checkpoint_interval'] == 'daily':
            self.checkpoint_interval = self.dt * 24
        elif self.config['simulation']['checkpoint_interval'] == 'weekly':
            self.checkpoint_interval = self.dt * 24 * 7

        self.version = self.config['simulation']['named_version']
        self.set_run_dir()

        if self.config['simulation']['run_rbo_mpc']:
            # Run baseline MPC with N hour horizon, no aggregator
            # Run baseline with 1 hour horizon for non-MPC HEMS
            self.case = "baseline" # no aggregator level control
            # for self.mpc in self.mpc_permutations:
            # for self.version in self.versions:
            self.flush_redis()
            self.get_homes()
            self.reset_collected_data()
            self.run_baseline()
            self.write_outputs()

if __name__=="__main__":
    a = Aggregator()
    a.run()
