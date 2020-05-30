import os
import sys
import threading
from queue import Queue

import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
import random
import names
from redis import StrictRedis
import string
import cvxpy as cp
import dccp
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# Local
from dragg.aggregator_logger import AggregatorLogger
from dragg.mpc_calc import MPCCalc


class Aggregator:
    def __init__(self):
        self.agg_log = AggregatorLogger()
        self.data_dir = 'data'
        self.outputs_dir = 'outputs'
        if not os.path.isdir(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.json'))
        self.ts_data_file = os.path.join(self.data_dir, os.environ.get('SOLAR_TEMPERATURE_DATA_FILE', 'nsrdb.csv'))
        self.tou_data_file = os.path.join(self.data_dir, os.environ.get('TOU_DATA_FILE', 'tou_data.xlsx'))
        self.required_keys = {
            "total_number_homes",
            "homes_battery",
            "homes_pv",
            "homes_pv_battery",
            "home_hvac_r_dist",
            "home_hvac_c_dist",
            "home_hvac_p_cool_dist",
            "home_hvac_p_heat_dist",
            "wh_r_dist",
            "wh_c_dist",
            "wh_p_dist",
            "battery_max_rate",
            "battery_capacity",
            "battery_cap_bounds",
            "battery_charge_eff",
            "battery_discharge_eff",
            "pv_area",
            "pv_efficiency",
            "start_datetime",
            "end_datetime",
            "prediction_horizons",
            "random_seed",
            "load_zone",
            "step_size_coeff",
            "max_load_threshold",
            "check_type"
        }
        self.timestep = None  # Set by redis_set_initial_values
        self.iteration = None  # Set by redis_set_initial_values
        self.reward_price = None  # Set by redis_set_initial_values
        self.start_hour_index = None  # Set by calc_star_hour_index
        # self.horizon = None  # Set by redis_set_initial_values
        # self.agg_load = None  # Set after every iteration
        self.agg_load = 0
        self.baseline_data = {}
        self.baseline_agg_load_list = []  # Aggregate load at every timestep from the baseline run
        self.max_agg_load = None  # Set after baseline run, the maximum aggregate load over all the timesteps
        self.max_agg_load_list = []
        # self.max_agg_load_threshold = None  # Set after baseline run, max_agg_load * threshold value set
        # self.max_agg_load_threshold_list = []
        self.converged = False
        self.num_threads = 1
        self.start_dt = None  # Set by _set_dt
        self.end_dt = None  # Set by _set_dt
        self.hours = None  # Set by _set_dt
        self.all_homes = None  # Set by create_homes
        self.queue = Queue()
        self.redis_client = StrictRedis(host=os.environ.get('REDIS_HOST'), decode_responses=True)
        self.config = self._import_config()
        self.step_size_coeff = self.config["step_size_coeff"]
        self.check_type = self.config["check_type"]  # One of: 'pv_only', 'base', 'battery_only', 'pv_battery', 'all'
        self.ts_data = self._import_ts_data()  # Temp: degC, RH: %, Pressure: mbar, GHI: W/m2
        self.tou_data = self._import_tou_data()  # SPP: $/kWh
        self.all_data = self.join_data()
        self._set_dt()
        self._build_tou_price()
        self.all_data.drop("ts", axis=1)

        self.all_rps = np.zeros(self.hours)
        self.all_sps = np.zeros(self.hours)

        self.agg_setpoint = None
        self.action = 0
        self.state = 0
        self.epsilon = float(self.config["agg_exploration_rate"])
        self.alpha = float(self.config["agg_learning_rate"])
        self.lam = float(self.config["rl_agg_regularization_factor"])
        self.theta = np.vstack(np.zeros(6))
        self.beta = float(self.config["rl_agg_discount_factor"])

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.agg_log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)
        with open(self.config_file, 'r') as f:
            data = json.load(f)
            d_keys = set(data.keys())
            is_subset = self.required_keys.issubset(d_keys)
            if not is_subset:
                self.agg_log.logger.error(f"Not all required keys specified in config file. These must be specified: {self.required_keys}")
                sys.exit(1)
            return data

    def _set_dt(self):
        """
        Convert the start and end datetimes specified in the config file into python datetime
        objects.  Calculate the number of hours for which the simulation will run.
        :return:
        """
        try:
            self.start_dt = datetime.strptime(self.config["start_datetime"], '%Y-%m-%d %H')
            self.end_dt = datetime.strptime(self.config["end_datetime"], '%Y-%m-%d %H')
        except ValueError as e:
            self.agg_log.logger.error(f"Error parsing datetimes: {e}")
            sys.exit(1)
        self.hours = self.end_dt - self.start_dt
        self.hours = int(self.hours.total_seconds() / 3600)
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
        df = df[df["Minute"] == 0]
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
        if not os.path.exists(self.tou_data_file):
            self.agg_log.logger.error(f"TOU data file does not exist: {self.tou_data_file}")
            sys.exit(1)
        df_all = pd.read_excel(self.tou_data_file, sheet_name=None)
        k1 = list(df_all.keys())[0]
        df = df_all[k1]
        for k, v in df_all.items():
            if k == k1:
                pass
            else:
                df = df.append(v, ignore_index=True)

        df = df[df["Settlement Point"] == self.config["load_zone"]]
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
        sd_times = self.config["shoulder_times"]
        pk_times = self.config["peak_times"]
        op_price = float(self.config["offpeak_price"])
        sd_price = float(self.config["shoulder_price"])
        pk_price = float(self.config["peak_price"])
        self.all_data['tou'] = self.all_data['ts'].apply(lambda x: pk_price if (x.hour <= pk_times[1] and x.hour >= pk_times[0]) else (sd_price if x.hour <= sd_times[1] and x.hour >= sd_times[0] else op_price))

    def join_data(self):
        """
        Join the TOU, GHI, temp data into a single dataframe
        :return: pandas.DataFrame
        """
        df = pd.merge(self.ts_data, self.tou_data, on='ts')
        return df.set_index('ts', drop=False)

    def _check_home_configs(self):
        base_homes = [e for e in self.all_homes if e["type"] == "base"]
        pv_battery_homes = [e for e in self.all_homes if e["type"] == "pv_battery"]
        pv_only_homes = [e for e in self.all_homes if e["type"] == "pv_only"]
        battery_only_homes = [e for e in self.all_homes if e["type"] == "battery_only"]
        if not len(base_homes) == self.config["total_number_homes"] - self.config["homes_battery"] - self.config["homes_pv"] - self.config["homes_pv_battery"]:
            self.agg_log.logger.error("Incorrect number of base homes.")
            sys.exit(1)
        elif not len(pv_battery_homes) == self.config["homes_pv_battery"]:
            self.agg_log.logger.error("Incorrect number of base pv_battery homes.")
            sys.exit(1)
        elif not len(pv_only_homes) == self.config["homes_pv"]:
            self.agg_log.logger.error("Incorrect number of base pv_only homes.")
            sys.exit(1)
        elif not len(battery_only_homes) == self.config["homes_battery"]:
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
        self.config["random_seed"] = new_seed

    def create_homes(self):
        """
        Given parameter distributions and number of homes of each type, create a list
        of dictionaries of homes with the parameters set for each home.
        :return:
        """
        # Set seed before sampling.  Will ensure home name and parameters
        # are the same throughout different runs
        np.random.seed(self.config["random_seed"])
        random.seed(self.config["random_seed"])

        # Define home and HVAC parameters
        home_r_dist = np.random.uniform(
            self.config["home_hvac_r_dist"][0],
            self.config["home_hvac_r_dist"][1],
            self.config["total_number_homes"]
        )
        home_c_dist = np.random.uniform(
            self.config["home_hvac_c_dist"][0],
            self.config["home_hvac_c_dist"][1],
            self.config["total_number_homes"]
        )
        home_hvac_p_cool_dist = np.random.uniform(
            self.config["home_hvac_p_cool_dist"][0],
            self.config["home_hvac_p_cool_dist"][1],
            self.config["total_number_homes"]
        )
        home_hvac_p_heat_dist = np.random.uniform(
            self.config["home_hvac_p_heat_dist"][0],
            self.config["home_hvac_p_heat_dist"][1],
            self.config["total_number_homes"]
        )

        # Define water heater parameters
        wh_r_dist = np.random.uniform(
            self.config["wh_r_dist"][0],
            self.config["wh_r_dist"][1],
            self.config["total_number_homes"]
        )
        wh_c_dist = np.random.uniform(
            self.config["wh_c_dist"][0],
            self.config["wh_c_dist"][1],
            self.config["total_number_homes"]
        )
        wh_p_dist = np.random.uniform(
            self.config["wh_p_dist"][0],
            self.config["wh_p_dist"][1],
            self.config["total_number_homes"]
        )

        alpha_dist = np.random.uniform(
            self.config["alpha_beta_dist"][0],
            self.config["alpha_beta_dist"][1],
            self.config["total_number_homes"]
        )
        beta_dist = np.random.uniform(
            self.config["alpha_beta_dist"][0],
            self.config["alpha_beta_dist"][1],
            self.config["total_number_homes"]
        )
        all_homes = []

        # PV values are constant
        pv = {
            "area": self.config["pv_area"],
            "eff": self.config["pv_efficiency"]
        }

        # battery values also constant
        battery = {
            "max_rate": self.config["battery_max_rate"],
            "capacity": self.config["battery_capacity"],
            "capacity_lower": self.config["battery_cap_bounds"][0] * self.config["battery_capacity"],
            "capacity_upper": self.config["battery_cap_bounds"][1] * self.config["battery_capacity"],
            "ch_eff": self.config["battery_charge_eff"],
            "disch_eff": self.config["battery_discharge_eff"],
            "batt_cons": self.config["battery_cons_penalty"]
        }

        i = 0
        # Define pv and battery homes
        for _ in range(self.config["homes_pv_battery"]):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "pv_battery",
                "alpha": alpha_dist[i],
                "beta": beta_dist[i],
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i]
                },
                "battery": battery,
                "pv": pv
            })
            i += 1

        # Define pv only homes
        for _ in range(self.config["homes_pv"]):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "pv_only",
                "alpha": alpha_dist[i],
                "beta": beta_dist[i],
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i]
                },
                "pv": pv
            })
            i += 1

        # Define battery only homes
        for _ in range(self.config["homes_battery"]):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "battery_only",
                "alpha": alpha_dist[i],
                "beta": beta_dist[i],
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i]
                },
                "battery": battery
            })
            i += 1

        base_homes = self.config["total_number_homes"] - self.config["homes_battery"] - self.config["homes_pv"] - self.config["homes_pv_battery"]
        for _ in range(base_homes):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            all_homes.append({
                "name": names.get_first_name() + '-' + res,
                "type": "base",
                "alpha": alpha_dist[i],
                "beta": beta_dist[i],
                "hvac": {
                    "r": home_r_dist[i],
                    "c": home_c_dist[i],
                    "p_c": home_hvac_p_cool_dist[i],
                    "p_h": home_hvac_p_heat_dist[i]
                },
                "wh": {
                    "r": wh_r_dist[i],
                    "c": wh_c_dist[i],
                    "p": wh_p_dist[i]
                }
            })
            i += 1

        self.all_homes = all_homes
        self._check_home_configs()

    def reset_baseline_data(self):
        self.baseline_agg_load_list = []
        for home in self.all_homes:
            self.baseline_data[home["name"]] = {
                "type": home["type"],
                "baseline_price": [],
                "reward_price": [],
                "temp_in_opt": [],
                "temp_wh_opt": [],
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
                self.baseline_data[home["name"]]["e_batt_opt"] = []
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
        if not self.end_dt + timedelta(hours=max(self.config["prediction_horizons"])) <= self.all_data.index[-1]:
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
        self.reward_price = 0
        temp_sp = self.config["temp_sp"]
        wh_sp = self.config["wh_sp"]
        min_runtime = self.config["min_runtime_mins"]
        self.e_batt_init = self.config["battery_capacity"] * self.config["battery_cap_bounds"][0]
        self.redis_client.hset("initial_values", "temp_in_init", self.config["temp_in_init"])
        self.redis_client.hset("initial_values", "temp_wh_init", self.config["temp_wh_init"])
        self.redis_client.hset("initial_values", "e_batt_init", self.e_batt_init)
        self.redis_client.hset("initial_values", "temp_in_min", temp_sp[0])
        self.redis_client.hset("initial_values", "temp_in_max", temp_sp[1])
        self.redis_client.hset("initial_values", "temp_wh_min", wh_sp[0])
        self.redis_client.hset("initial_values", "temp_wh_max", wh_sp[1])
        self.redis_client.hset("initial_values", "min_runtime_mins", min_runtime)
        self.redis_client.set("start_hour_index", self.start_hour_index)
        self.redis_client.hset("current_values", "timestep", self.timestep)
        self.redis_client.hset("current_values", "reward_price", self.reward_price)

        if self.case == "agg_mpc":
            self.iteration = 0
            self.redis_client.hset("current_values", "iteration", self.iteration)

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
                    self.redis_client.hset(home, k, v[-1])
                elif k == "temp_wh_opt":
                    self.redis_client.hset(home, k, v[-1])
                if 'battery' in vals['type'] and k == "e_batt_opt":
                    self.redis_client.hset(home, k, v[-1])

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
                self.redis_client.rpush(c, val)

    def redis_set_current_values(self):
        self.redis_client.hset("current_values", "timestep", self.timestep)
        self.redis_client.hset("current_values", "reward_price", float(self.reward_price))
        self.all_rps[self.timestep] = self.reward_price

        if self.case == "agg_mpc":
            self.redis_client.hset("current_values", "iteration", self.iteration)
        elif self.case == "rl_agg":
            self.all_sps[self.timestep] = self.agg_setpoint

    def update_reward_price(self):
        rp = self.reward_price + self.step_size_coeff * self.marginal_demand
        return rp

    def _calc_state(self):
        # xk = (self.agg_load - self.agg_setpoint)/self.agg_setpoint
        xk = (self.agg_load-self.agg_setpoint)/self.agg_setpoint
        return xk

    def _cost(self, xk, xk1):
        sigma = 0.1
        mu = 0
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2)

    def _q(self, state, action):
        q = np.matmul(self.theta.T, self._phi(state, action))
        return q

    def _phi(self, state, action):
        # phi = np.array([1, state, action, state*action, state**2, action**2])
        phi = np.array([1, state, action, 0, 0, 0])
        return phi

    def _qvalue(self):
        q_k = self._cost(self.state, self.next_state) + self.beta * self._q(self.next_state, self._get_greedyaction(self.next_state))
        return q_k

    def _get_greedyaction(self, state_k):
        try:
            u_k = cp.Variable(1)
            xhat_k = cp.Constant(state_k)
            th = cp.Constant(self.theta)
            t = cp.Variable(5)
            cons = [u_k >= self.actionspace[0],
                    u_k <= self.actionspace[1],
                    t[0] == self.theta[0],
                    t[1] == self.theta[1]*xhat_k,
                    t[2] == self.theta[2]*u_k,
                    t[3] == self.theta[3]*xhat_k*u_k,
                    t[4] == self.theta[4]*xhat_k**2]

            obj = cp.Minimize(cp.sum(t) + self.theta[5]*(u_k**2))
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.ECOS, method="dccp", verbose=False)
            uk = u_k.value[0]
        except:
            uk = 0
            self.greedy_failed = True
        return uk

    def update_qfunction(self):
        # self.q_k = np.array(self._qvalue())
        # self.phi_k = self._phi(self.state, self.action)
        # self.phi_k1 = self._phi(self.next_state, self._get_greedyaction(self.next_state))
        #
        # if self.timestep == 0: # initialize self.phi and self.q
        #     self.q = self.q_k
        #     self.phi = self.phi_k
        # else: # tack on new observed values
        #     self.q = np.vstack((self.q, self.q_k))
        #     self.phi = np.vstack((self.phi, self.phi_k))
        #
        # if self.timestep > 24: # remove old observation values
        #     self.q = self.q[1:,]
        #     self.phi = self.phi[1:,]
        #
        # # # Ridge Regression
        # # phi_sq = np.matmul(np.vstack(self.phi).T, np.vstack(self.phi))
        # # state_inv = np.asmatrix(phi_sq + self.lam * np.eye(6)).I
        # # state_inv = np.matmul(state_inv, np.asmatrix(self.phi).T)
        # # self.theta_k = np.matmul(state_inv, np.asmatrix(self.q))
        #
        # # # Lasso Regression
        # # clf = Lasso()
        # # clf.fit(self.phi.reshape((-1,6)), self.q.reshape((-1,1)))
        # # self.theta_k = np.array(clf.coef_)
        #
        # # # Elastic Net Regression
        # # clf = ElasticNet()
        # # clf.fit(self.phi.reshape((-1,6)), self.q.reshape((-1,1)))
        # # self.theta_k = np.array(clf.coef_)
        # #
        # # self.theta = self.theta_k.flatten()
        # # # self.theta = (1-self.alpha)*self.theta + self.alpha*(self.theta_k)
        # # # self.theta = self.theta + self.alpha*(self.theta_k - self.theta)
        # # self.theta.tolist()

        # self.q_k = self._qvalue()
        self.theta = self.theta.flatten()
        self.phi_k = (self._phi(self.state, self.action))
        next_action = self._get_greedyaction(self.next_state)
        self.phi_k1 = (self._phi(self.next_state, next_action))
        self.q = self._qvalue()
        if self.timestep > 10:
            self.theta = self.theta + self.alpha * (self._q(self.state, self.action) - self._cost(self.next_state) - self.beta*self._q(self.next_state, next_action))*np.transpose(self.phi_k - self.phi_k1)
        # self.theta = self.theta - self.alpha*np.transpose(self.phi_k - self.phi_k1)

    def rl_update_reward_price(self):
        self.actionspace = self.config["action_space"]
        if np.random.uniform(0,1) >= self.epsilon: # the greedy action
            u_k = self._get_greedyaction(self.state)
            self.is_greedy = True
        else: # exploration
            u_k = random.uniform(self.actionspace[0], self.actionspace[1])
            self.is_greedy = False
        self.action = np.round(u_k, 2)
        # self.action = u_k

        self.reward_price = self.action

    def set_baseline_initial_vals(self):
        for home in self.all_homes:
            self.baseline_data[home["name"]]["temp_in_opt"].append(self.config["temp_in_init"])
            self.baseline_data[home["name"]]["temp_wh_opt"].append(self.config["temp_wh_init"])
            if 'battery' in home["type"]:
                self.baseline_data[home["name"]]["e_batt_opt"].append(self.config["battery_cap_bounds"][0] * self.config["battery_capacity"])

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

    def run_iteration(self, horizon=1):
        worker = MPCCalc(self.queue, horizon)
        worker.run()

        # Block in Queue until all tasks are done
        self.queue.join()

        self.agg_log.logger.info("Workers complete")
        self.agg_log.logger.info(f"Number of threads: {threading.active_count()}")
        self.agg_log.logger.info(f"Length of queue: {self.queue.qsize()}")

    def collect_data(self):
        agg_load = 0
        agg_cost = 0
        for home in self.all_homes:
            if self.check_type == 'all' or home["type"] == self.check_type:
                vals = self.redis_client.hgetall(home["name"])
                for k, v in vals.items():
                    self.baseline_data[home["name"]][k].append(float(v))
                agg_load += float(vals["p_grid_opt"])
                agg_cost += float(vals["cost_opt"])
        self.agg_load = agg_load
        self.agg_cost = agg_cost
        self.baseline_agg_load_list.append(agg_load)

    def collect_fake_data(self):
        self.baseline_agg_load_list.append(self.agg_load)

    def check_agg_mpc_data(self):
        self.agg_load = 0
        self.agg_cost = 0
        for home in self.all_homes:
            if self.check_type == 'all' or home["type"] == self.check_type:
                vals = self.redis_client.hgetall(home["name"])
                self.agg_load += float(vals["p_grid_opt"])
                self.agg_cost += float(vals["cost_opt"])
        self.marginal_demand = max(self.agg_load - self.max_load_threshold, 0)
        self.agg_log.logger.info(f"Aggregate Load: {self.agg_load:.20f}")
        self.agg_log.logger.info(f"Max Threshold: {self.max_load_threshold:.20f}")
        self.agg_log.logger.info(f"Marginal Demand: {self.marginal_demand:.20f}")
        if self.marginal_demand == 0:
            self.converged = True

    def update_agg_mpc_data(self):
        self.agg_mpc_data[self.timestep]["reward_price"].append(self.reward_price)
        self.agg_mpc_data[self.timestep]["agg_cost"].append(self.agg_cost)
        self.agg_mpc_data[self.timestep]["agg_load"].append(self.agg_load)

    def record_rl_agg_data(self):
        # self.marginal_demand = max(self.agg_load - self.max_load_threshold, 0)
        # self.baseline_agg_load_list.append(self.agg_load)
        self.agg_log.logger.info(f"Aggregate Load: {self.agg_load:.20f}")
        self.agg_log.logger.info(f"Desired Setpoint: {self.agg_setpoint:.20f}")
        # self.agg_log.logger.info(f"Marginal Demand: {self.marginal_demand:.20f}") # don't need but should we record this anyways?
        self.rl_agg_data[self.timestep]["reward_price"] = float(self.reward_price)
        self.rl_agg_data[self.timestep]["agg_cost"] = self.agg_cost
        self.rl_agg_data[self.timestep]["agg_load"] = self.agg_load
        self.rl_agg_data[self.timestep]["is_greedy"] = self.is_greedy
        self.rl_agg_data[self.timestep]["greedy_failed"] = self.greedy_failed

    def record_rl_q_data(self):
        self.rl_q_data["timestep"].append(self.timestep)
        self.rl_q_data["theta"].append(self.theta.flatten().tolist())
        # self.rl_q_data["theta_k"].append(self.theta_k.flatten().tolist())
        self.rl_q_data["phi"].append(self.phi_k.tolist())
        # self.rl_q_data["q"].append(self.q_k.tolist())
        self.rl_q_data["q"].append(self.q)
        self.rl_q_data["action"].append(self.action.tolist())
        self.rl_q_data["state"].append(self.state)
        self.rl_q_data["is_greedy"].append(self.is_greedy)
        self.rl_q_data["greedy_failed"].append(self.greedy_failed)

    def run_baseline(self, horizon=1):
        self.agg_log.logger.info(f"Performing baseline run for horizon: {horizon}")
        self.start_time = datetime.now()
        for hour in range(self.hours):
            for home in self.all_homes:
                if self.check_type == "all" or home["type"] == self.check_type:
                    self.queue.put(home)
            self.redis_set_current_values()
            self.run_iteration(horizon)
            self.collect_data()
            self.timestep += 1
        # Write
        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {horizon}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")
        self.check_baseline_vals()

    def summarize_baseline(self, horizon=1):
        """
        Get the maximum of the aggregate demand
        :return:
        """
        self.max_agg_load = max(self.baseline_agg_load_list)
        # self.max_agg_load_threshold = self.max_agg_load * self.max_load_threshold
        self.max_agg_load_list.append(self.max_agg_load)
        # self.max_agg_load_threshold_list.append(self.max_agg_load_threshold)

        self.agg_log.logger.info(f"Max load list: {self.max_agg_load_list}")
        self.baseline_data["Summary"] = {
            "case": self.case,
            "start_datetime": self.start_dt.strftime('%Y-%m-%d %H'),
            "end_datetime": self.end_dt.strftime('%Y-%m-%d %H'),
            "solve_time": self.t_diff.total_seconds(),
            "horizon": horizon,
            "num_homes": self.config['total_number_homes'],
            "p_max_aggregate": self.max_agg_load,
            # "p_max_aggregate_threshold": self.max_agg_load_threshold,
            "p_grid_aggregate": self.baseline_agg_load_list,
            "SPP": self.all_data.loc[self.mask, "SPP"].values.tolist(),
            "OAT": self.all_data.loc[self.mask, "OAT"].values.tolist(),
            "GHI": self.all_data.loc[self.mask, "GHI"].values.tolist(),
            "TOU": self.all_data.loc[self.mask, "tou"].values.tolist(),
            "RP": self.all_rps.tolist(),
            "p_grid_setpoint": self.all_sps.tolist()
        }

    def write_outputs(self, horizon):
        # Write values for baseline run to file
        case_output = os.path.join(self.outputs_dir, self.case)
        if not os.path.isdir(case_output):
            os.makedirs(case_output)
        file = os.path.join(case_output, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}-{self.case}_{self.check_type}-homes_{self.config['total_number_homes']}-horizon_{horizon}-results.json")
        with open(file, 'w+') as f:
            json.dump(self.baseline_data, f, indent=4)
        if self.case == "agg_mpc":
            f2 = os.path.join(case_output, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}-{self.case}_{self.check_type}-homes_{self.config['total_number_homes']}-horizon_{horizon}-iter-results.json")
            with open(f2, 'w+') as f:
                json.dump(self.agg_mpc_data, f, indent=4)
        if self.case == "rl_agg":
            f3 = os.path.join(case_output, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}-{self.case}_{self.check_type}-homes_{self.config['total_number_homes']}-horizon_{horizon}-iter-results.json")
            with open(f3, 'w+') as f:
                json.dump(self.rl_agg_data, f, indent=4)
            f4 = os.path.join(case_output, f"{self.start_dt.strftime('%Y-%m-%dT%H')}_{self.end_dt.strftime('%Y-%m-%dT%H')}-{self.case}_{self.check_type}-homes_{self.config['total_number_homes']}-horizon_{horizon}-q-results.json")
            with open(f4, 'w+') as f:
                json.dump(self.rl_q_data, f, indent=4)

    def write_home_configs(self):
        # Write all home configurations to file
        ah = os.path.join(self.outputs_dir, f"all_homes-{self.config['total_number_homes']}-config.json")
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

    def run_agg_mpc(self, horizon):
        self.agg_log.logger.info(f"Performing AGG MPC run for horizon: {horizon}")
        self.start_time = datetime.now()
        self.agg_mpc_data = self.set_agg_mpc_initial_vals()
        for hour in range(self.hours):
            self.converged = False
            while True:
                for home in self.all_homes:
                    if self.check_type == "all" or home["type"] == self.check_type:
                        self.queue.put(home)
                self.redis_set_current_values()
                self.run_iteration(horizon)
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
            self.timestep += 1
        # Write
        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {horizon}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")
        self.check_baseline_vals()

    def set_rl_agg_initial_vals(self): # initializes an empty list of important data the length of the simulation
        self.q = np.array([])
        temp = []
        for h in range(self.hours):
            temp.append({
                "timestep": h,
                "reward_price": None,
                "agg_cost": None,
                "agg_load": None
            })
        return temp

    def set_rl_q_initial_vals(self):
        temp = {}
        temp["timestep"] = [-1]
        temp["theta"] = []
        temp["phi"] = []
        temp["q"] = []
        temp["action"] = []
        temp["state"] = [0]
        temp["is_greedy"] = []
        temp["greedy_failed"] = []
        return temp

    def _read_setpoint(self):
        baseline = os.path.join(self.outputs_dir, "baseline", "2015-01-01T00_2015-01-10T00-baseline_all-homes_20-horizon_8-results.json")
        with open(baseline, 'r') as f:
            data = json.load(f)
        sp = np.array(data["Summary"]["p_grid_aggregate"])
        noise = 0.5*np.random.randn(9)*sp[1::24]
        noise = noise.repeat(24)
        sp += noise
        return sp

    def _gen_setpoint(self):
        # min_p_grid = 10
        # max_p_grid = 60
        # daily_offset = 6
        # sp = (max_p_grid - min_p_grid)*np.sin((self.timestep-6) * 3.14/12) + (min_p_grid + max_p_grid)/2
        # return sp
        # i = self.timestep % 24
        # if i >= 12:
        #     sp = 60
        # else:
        #     sp = 10
        sp = 40
        return sp

    def test_response(self):
        if self.timestep == 0:
            self.agg_load = 40
        self.agg_load += self.agg_load*0.001
        self.agg_cost = self.agg_load * self.reward_price

    def run_rl_agg(self, alpha, epsilon, horizon):
        self.agg_log.logger.info(f"Performing RL AGG (learning rate: {alpha}, exploration rate: {epsilon}) with MPC HEMS for horizon: {horizon}")
        self.start_time = datetime.now()
        self.rl_agg_data = self.set_rl_agg_initial_vals()
        self.rl_q_data = self.set_rl_q_initial_vals()
        self.baseline_agg_load_list = [0]
        self.state = 0
        for hour in range(self.hours):
            self.greedy_failed = False
            self.agg_setpoint = self._gen_setpoint()

            self.rl_update_reward_price()
            self.redis_set_current_values() # broadcast rl price to community

            for home in self.all_homes:
                if self.check_type == "all" or home["type"] == self.check_type:
                    self.queue.put(home)
            self.run_iteration(horizon) # community response to broadcasted price (done in a single iteration)
            self.collect_data()
            # self.test_response()
            # self.collect_fake_data()

            self.record_rl_agg_data() # record response to the broadcasted price
            self.next_state = self._calc_state() # this is the state at t = k+1
            # self.collect_data()
            self.update_qfunction()
            self.record_rl_q_data()

            self.timestep += 1
            self.state = self.next_state


        self.end_time = datetime.now()
        self.t_diff = self.end_time - self.start_time
        self.agg_log.logger.info(f"Horizon: {horizon}; Num Hours Simulated: {self.hours}; Run time: {self.t_diff.total_seconds()} seconds")

    def flush_redis(self):
        self.redis_client.flushall()
        self.agg_log.logger.info("Flushing Redis")
        time.sleep(1)
        self.check_all_data_indices()
        self.calc_start_hour_index()
        self.redis_add_all_data()

    def run(self):
        self.agg_log.logger.info("Made it to Aggregator Run")
        self.create_homes()
        self.write_home_configs()

        if self.config["run_baseline"]:
            # Run baseline - no MPC, no aggregator
            self.flush_redis()
            self.case = "baseline" # no aggregator
            horizon = 1
            self.redis_set_initial_values()
            self.reset_baseline_data()
            self.set_baseline_initial_vals()
            self.run_baseline(horizon)
            self.summarize_baseline(horizon)
            self.write_outputs(horizon)

        if self.config["run_rbo_mpc"]:
            # Run baseline MPC with N hour horizon, no aggregator
            self.case = "baseline" # no aggregator
            for h in self.config["prediction_horizons"]:
                self.flush_redis()
                self.redis_set_initial_values()
                self.reset_baseline_data()
                self.set_baseline_initial_vals()
                self.run_baseline(h)
                self.summarize_baseline(h)
                self.write_outputs(h)

        if self.config["run_agg_mpc"]:
            self.case = "agg_mpc"
            horizon = self.config["agg_mpc_horizon"]
            for threshold in self.config["max_load_threshold"]:
                self.max_load_threshold = threshold
                self.flush_redis()
                self.redis_set_initial_values()
                self.reset_baseline_data()
                self.set_baseline_initial_vals()
                self.run_agg_mpc(horizon)
                self.summarize_baseline(horizon)
                self.write_outputs(horizon)

        if self.config["run_rl_agg"]:
            self.case = "rl_agg"
            horizon = self.config["agg_mpc_horizon"]
            self.max_load_threshold = self.config["max_load_threshold"][0]
            params = [0]
            for param in params:
                self.flush_redis()
                self.redis_set_initial_values()
                self.reset_baseline_data()
                self.set_baseline_initial_vals()
                self.run_rl_agg(self.config["agg_learning_rate"], self.config["agg_exploration_rate"], horizon)
                self.summarize_baseline(horizon)
                self.write_outputs(horizon)
        #     self.redis_set_initial_values()
        #     self.run_mpc(h)
