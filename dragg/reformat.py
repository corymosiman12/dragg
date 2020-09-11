import os
import sys
import json
import toml
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import itertools as it
import random

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly

from dragg.logger import Logger

class Reformat:
    def __init__(self, add_outputs={}, agg_params={"rl_horizon":[1]}, mpc_params={}, versions=set([0.0]), date_ranges={"end_datetime":[]}, include_runs={}, log=Logger("reformat")):
        self.ref_log = log
        self.data_dir = os.path.expanduser(os.environ.get('DATA_DIR',' data'))
        # self.outputs_dir = set()
        self.outputs_dir = {"outputs"}
        if len(self.outputs_dir) == 0:
            self.ref_log.logger.error("No outputs directory found.")
            quit()
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
        self.config = self._import_config()

        self.include_runs = include_runs
        self.versions = versions

        self.date_folders = self.set_date_folders(date_ranges)
        self.mpc_folders = self.set_mpc_folders(mpc_params)
        self.baselines = self.set_base_file()
        self.parametrics = []
        self.parametrics = self.set_parametric_files(agg_params)

        np.random.seed(self.config['simulation']['random_seed'])
        self.fig_list = None
        self.save_path = os.path.join('outputs', 'images', datetime.now().strftime("%m%dT%H%M%S"))

    def main(self):
        if self.config['simulation']['run_rl_agg'] or self.config['simulation']['run_rbo_mpc']:
            # put a list of plotting functions here
            self.sample_home = "Crystal-RXXFA"
            self.plots = [self.rl2baseline,
                        self.rl2baseline_error,
                        self.plot_single_home]

        if self.config['simulation']['run_rl_simplified']:
            # put a list of plotting functions here
            self.plots = [self.rl_simplified,
                        self.rl_simplified_rp,
                        self.all_rps]

        self.images = self.plot_all()

    def tf_main(self):
        """ Intended for plotting an image suite for use with the tensorflow reinforcement learning package. """
        self.sample_home = "Lillie-NMHUH"
        self.plots = [self.rl2baseline,
                    self.rl2baseline_error,
                    self.plot_single_home]

        self.images = self.plot_all()

    def plot_all(self, save_images=False):
        figs = []
        for plot in self.plots:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.update_layout(
                font=dict(
                    # family="Courier New, monospace",
                    size=22,
                )
            )
            fig = plot(fig)
            fig.show()
            figs += [fig]
        return figs

    def create_summary(self, file):
        with open(file) as f:
            data = json.load(f)

        p_grid_agg = []
        for k,v in data.items():
            p_grid_agg.append(v['p_grid_opt'])
        p_grid_agg = np.sum(p_grid_agg, axis=0).tolist()

        p_grid_setpoint = (np.ones(len(p_grid_agg)-1) * self.config['community']['total_number_homes'][0] * self.config['community']['house_p_avg']).tolist()

        summary = {"p_grid_aggregate": p_grid_agg, "p_grid_setpoint": p_grid_setpoint}
        data["Summary"] = summary

        with open(file, 'w+') as f:
            json.dump(data, f, indent=4)

        return data

    def save_images(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        for img in self.images:
            self.ref_log.logger.info(f"Saving images of outputs to timestamped folder at {self.save_path}.")
            try:
                path = os.path.join(self.save_path, f"{img.layout.title.text}.png")
                pio.write_image(img, path, width=1024, height=768)
            except:
                self.ref_log.logger.error("Could not save plotly image(s) to outputs directory.")

    def add_date_ranges(self, additional_params):
        start_dates = [datetime.strptime(self.config['simulation']['start_datetime'], '%Y-%m-%d %H')]
        end_dates = set([datetime.strptime(self.config['simulation']['end_datetime'], '%Y-%m-%d %H')])
        temp = {"start_datetime": start_dates, "end_datetime": end_dates}
        for key in temp:
            if key in additional_params:
                for i in additional_params[key]:
                    temp[key].add(datetime.strptime(i, '%Y-%m-%d %H'))
        self.date_ranges = temp

    def add_agg_params(self, additional_params):
        alphas = set(self.config['rl']['parameters']['learning_rate'])
        epsilons = set(self.config['rl']['parameters']['exploration_rate'])
        betas = set(self.config['rl']['parameters']['discount_factor'])
        batch_sizes = set(self.config['rl']['parameters']['batch_size'])
        rl_horizons = set(self.config['rl']['utility']['rl_agg_action_horizon'])
        rl_interval = set(self.config['rl']['utility']['hourly_steps'])
        temp = {"alpha": alphas, "epsilon": epsilons, "beta": betas, "batch_size": batch_sizes, "rl_horizon": rl_horizons, "rl_interval": rl_interval}
        for key in temp:
            if key in additional_params:
                temp[key] |= set(additional_params[key])
        self.agg_params = temp

    def add_mpc_params(self, additional_params):
        n_houses = self.config['community']['total_number_homes'][0]
        mpc_horizon = self.config['home']['hems']['prediction_horizon']
        dt = self.config['home']['hems']['sub_subhourly_steps']
        # for i in self.config['rl']['utility']['hourly_steps']:
        #     for j in self.config['home']['hems']['sub_subhourly_steps']:
        #         dt.append(60 // i // j)
        check_type = self.config['simulation']['check_type']
        temp = {"n_houses": set([n_houses]), "mpc_prediction_horizons": set(mpc_horizon), "mpc_hourly_steps": set(dt), "check_type": set([check_type])}
        for key in temp:
            if key in additional_params:
                temp[key] |= set(additional_params[key])
        self.mpc_params = temp

        self.versions |= set(self.config['rl']['version'])

    def set_date_folders(self, additional_params):
        self.add_date_ranges(additional_params)
        temp = []
        self.date_ranges['mpc_steps'] = set(self.config['home']['hems']['sub_subhourly_steps'])
        self.date_ranges['rl_steps'] = set(self.config['rl']['utility']['hourly_steps'])
        keys, values = zip(*self.date_ranges.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        permutations = sorted(permutations, key=lambda i: i['end_datetime'], reverse=True)
        for j in self.outputs_dir:
            for i in permutations:
                date_folder = os.path.join(j, f"{i['start_datetime'].strftime('%Y-%m-%dT%H')}_{i['end_datetime'].strftime('%Y-%m-%dT%H')}_{60 // i['rl_steps']}-{60 // i['rl_steps'] // i['mpc_steps']}")
                print(date_folder)
                if os.path.isdir(date_folder):
                    hours = i['end_datetime'] - i['start_datetime']
                    hours = int(hours.total_seconds() / 3600)

                    timesteps = hours * i['rl_steps']
                    print(hours, i['rl_steps'])
                    minutes = 60 // i['rl_steps']
                    x_lims = [i['start_datetime'] + timedelta(minutes=minutes*x) for x in range(timesteps)]

                    new_folder = {"folder": date_folder, "hours": hours, "start_dt": i['start_datetime'], "name": j+" ", "ts": timesteps, "x_lims": x_lims, "agg_dt": i['rl_steps']}
                    temp.append(new_folder)

        if len(temp) == 0:
            self.ref_log.logger.error("No files found for the date ranges specified.")
            exit()

        return temp

    def set_mpc_folders(self, additional_params):
        self.add_mpc_params(additional_params)
        temp = []
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        for j in self.date_folders:
            # for k in self.config['rl']['utility']['hourly_steps']:
            for i in permutations:
                mpc_folder = os.path.join(j["folder"], f"{i['check_type']}-homes_{i['n_houses']}-horizon_{i['mpc_prediction_horizons']}-interval_{60 // i['mpc_hourly_steps'] // j['agg_dt']}")
                if os.path.isdir(mpc_folder):
                    # timesteps = j['hours'] * i['mpc_hourly_steps'] * k
                    # minutes = 60 // i['mpc_hourly_steps'] // k
                    # x_lims = [j['start_dt'] + timedelta(minutes=minutes*x) for x in range(timesteps)]
                    name = j['name']
                    set = {'path': mpc_folder, 'dt': i['mpc_hourly_steps'], 'ts': j['ts'], 'x_lims': j['x_lims'], 'name': name, "agg_dt":j['agg_dt']}
                    if not mpc_folder in temp:
                        temp.append(set)
        for x in temp:
            print(x['path'])
        return temp

    def set_base_file(self):
        temp = []
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        for j in self.mpc_folders:
            path = j['path']
            for i in permutations:
                for k in self.versions:
                    file = os.path.join(path, "baseline", f"baseline_version-{k}-results.json")
                    self.ref_log.logger.debug(f"Looking for baseline file at {file}")
                    if os.path.isfile(file):
                        name = f"Baseline - {j['name']} - v{k}"
                        with open(file) as f:
                            data = json.load(f)
                        if "Summary" not in data:
                            self.create_summary(file)
                        set = {"results": file, "name": name, "parent": j}
                        temp.append(set)
                        self.ref_log.logger.info(f"Adding baseline file at {file}")

        return temp

    def set_rl_files(self, additional_params):
        temp = []
        names = []
        self.add_agg_params(additional_params)
        counter = 1
        for i in self.mpc_folders:
            path = i['path']
            rl_agg_folder = os.path.join(path, "rl_agg")
            all_params = {**self.agg_params, **self.mpc_params}
            keys, values = zip(*all_params.items())
            permutations = [dict(zip(keys, v)) for v in it.product(*values)]
            for j in permutations:
                for vers in self.versions:
                    if os.path.isdir(rl_agg_folder):
                        rl_agg_path = f"agg_horizon_{j['rl_horizon']}-interval_{60 // j['rl_interval']}-alpha_{j['alpha']}-epsilon_{j['epsilon']}-beta_{j['beta']}_batch-{j['batch_size']}_version-{vers}"
                        rl_agg_file = os.path.join(rl_agg_folder, rl_agg_path, "results.json")
                        self.ref_log.logger.debug(f"Looking for a RL aggregator file at {rl_agg_file}")
                        if os.path.isfile(rl_agg_file):
                            q_results = os.path.join(rl_agg_path, "q-results.json")
                            q_file = os.path.join(rl_agg_folder, q_results)
                            # name = i['name']
                            name = ""
                            # for k,v in j.items():
                            #     if len(all_params[k]) > 1 or k == "mpc_hourly_steps":
                            #         name += f"{k} = {v}, "
                            # name =  f"horizon={j['rl_horizon']}, alpha={j['alpha']}, beta={j['beta']}, epsilon={j['epsilon']}, batch={j['batch_size']}, disutil={j['mpc_disutility']}, discomf={j['mpc_discomfort']}"
                            name += f"v = {vers}"
                            with open(rl_agg_file) as f:
                                data = json.load(f)
                            if "Summary" not in data:
                                self.create_summary(rl_agg_file)
                            set = {"results": rl_agg_file, "q_results": q_file, "name": name, "parent": i, "rl_agg_action_horizon": j["rl_horizon"], "params": j, "skip": j['rl_interval'] // i['dt']}
                            if not name in names:
                                temp.append(set)
                                names.append(name)
                            self.ref_log.logger.info(f"Adding an RL aggregator agent file at {rl_agg_file}")

        if len(temp) == 0:
            self.ref_log.logger.warning("Parameterized RL aggregator runs are empty for this config file.")

        return temp

    def set_simplified_files(self, additional_params):
        temp = []
        self.add_agg_params(additional_params)
        for i in self.mpc_folders:
            path = i['path']
            simplified_folder = os.path.join(path, "simplified")
            all_params = {**self.agg_params, **self.mpc_params}
            keys, values = zip(*all_params.items())
            permutations = [dict(zip(keys, v)) for v in it.product(*values)]
            for j in permutations:
                for vers in self.versions:
                    if os.path.isdir(simplified_folder):
                        simplified_path = f"agg_horizon_{j['rl_horizon']}-alpha_{j['alpha']}-epsilon_{j['epsilon']}-beta_{j['beta']}_batch-{j['batch_size']}_version-{vers}"
                        simplified_file = os.path.join(simplified_folder, simplified_path, "results.json")
                        if os.path.isfile(simplified_file):
                            q_file = os.path.join(simplified_folder, simplified_path, "q-results.json")
                            if os.path.isfile(q_file):
                                # name = i['name']
                                name = ""
                                for k,v in j.items():
                                    if len(all_params[k]) > 1:
                                        name += f"{k} = {v}, "
                                set = {"results": simplified_file, "q_results": q_file, "name": name, "parent": i}
                                temp.append(set)

        return temp

    def set_parametric_files(self, additional_params):
        if self.config['simulation']['run_rl_agg'] or "rl_agg" in self.include_runs:
            self.parametrics += self.set_rl_files(additional_params)

        if self.config['simulation']['run_rl_simplified'] or "simplified" in self.include_runs:
            self.parametrics += self.set_simplified_files(additional_params)

        return self.parametrics

    def set_other_files(self, otherfile):
        self.parametrics.append(otherfile)

    def _type_list(self, type):
        type_list = set([])
        i = 0
        for file in (self.baselines + self.parametrics):
            with open(file["results"]) as f:
                data = json.load(f)

            temp = set([])
            for name, house in data.items():
                try:
                    if house["type"] == type:
                        temp.add(name)
                except:
                    pass

            if i < 1:
                type_list = temp
            else:
                type_list = type_list.intersection(temp)

        return type_list

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.ref_log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)

        with open(self.config_file, 'r') as f:
            data = toml.load(f)

        return data

    def plot_environmental_values(self, name, fig, summary, file, fname):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["OAT"][0:file["parent"]["ts"]], name=f"OAT (C)"))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["GHI"][0:file["parent"]["ts"]], name=f"GHI (W/m2)"))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["TOU"][0:file["parent"]["ts"]], name=f"TOU Price ($/kWh)", line_shape='hv'), secondary_y=True)
        fig = self.plot_thermal_bounds(fig, file['parent']['x_lims'], name, fname)
        return fig

    def plot_thermal_bounds(self, fig, x_lims, name, fname):
        for i in self.outputs_dir:
            ah_file = os.path.join(i, f"all_homes-{self.config['community']['total_number_homes'][0]}-config.json")
        with open(ah_file) as f:
            data = json.load(f)

        for dict in data:
            if dict['name'] == name:
                data = dict

        fig.add_trace(go.Scatter(x=x_lims, y=data['hvac']['temp_in_min'] * np.ones(len(x_lims)), name=f"Tin_min(C) - {fname}", fill=None, mode='lines', line_color='indigo'))
        fig.add_trace(go.Scatter(x=x_lims, y=data['hvac']['temp_in_max'] * np.ones(len(x_lims)), name=f"Tin_max - {fname}", fill='tonexty' , mode='lines', line_color='indigo'))

        fig.add_trace(go.Scatter(x=x_lims, y=data['wh']['temp_wh_min'] * np.ones(len(x_lims)), name=f"Twh_min(C) - {fname}", fill=None, mode='lines', line_color='red'))
        fig.add_trace(go.Scatter(x=x_lims, y=data['wh']['temp_wh_max'] * np.ones(len(x_lims)), name=f"Twh_max - {fname}", fill='tonexty' , mode='lines', line_color='red'))
        return fig

    def plot_base_home(self, name, fig, data, summary, fname, file, plot_price=True):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["temp_in_opt"], name=f"Tin (C) - {fname}"))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["temp_wh_opt"], name=f"Twh (C) - {fname}"))
        # self.plot_thermal_bounds(fig, file['parent']['x_lims'], name, fname)

        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_grid_opt"], name=f"Pgrid (kW) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_load_opt"], name=f"Pload (kW) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["hvac_cool_on_opt"], name=f"HVAC Cool Cmd - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["hvac_heat_on_opt"], name=f"HVAC Heat Cmd - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["wh_heat_on_opt"], name=f"WH Heat Cmd - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
        if plot_price:
            actual_price = np.add(summary["TOU"][:len(summary['RP'])], summary["RP"])
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=actual_price, name=f"Actual Price ($/kWh) - {fname}", visible='legendonly'), secondary_y=True)
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(actual_price), np.arange(len(actual_price))+1), name=f"Average Actual Price ($/kWh) - {fname}", visible='legendonly'), secondary_y=True)
        return fig

    def plot_pv(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_pv_opt"], name=f"Ppv (kW) - {fname}", line_shape='hv'))
        return fig

    def plot_battery(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["e_batt_opt"], name=f"SOC (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_batt_ch"], name=f"Pch (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_batt_disch"], name=f"Pdis (kW) - {fname}", line_shape='hv'))
        return fig

    def plot_single_home(self, fig):
        if self.sample_home is None:
            if type is None:
                type = "base"
                self.ref_log.logger.warning("Specify a home type or name. Proceeding with home of type: \"base\".")

            type_list = self._type_list(type)
            self.sample_home = random.sample(type_list,1)[0]
            self.ref_log.logger.info(f"Proceeding with home: {name}")

        flag = False
        for file in (self.baselines + self.parametrics):
            with open(file["results"]) as f:
                comm_data = json.load(f)

            try:
                data = comm_data[self.sample_home]
            except:
                self.ref_log.logger.error(f"No home with name: {self.sample_home}")
                return

            type = data["type"]
            summary = comm_data["Summary"]

            if not flag:
                fig = self.plot_environmental_values(self.sample_home, fig, summary, file, file["name"])
                flag = True

            fig = self.plot_base_home(self.sample_home, fig, data, summary, file["name"], file)

            fig.update_xaxes(title_text="Time of Day (hour)")
            fig.update_layout(title_text=f"{self.sample_home} - {type} type")

            if 'pv' in type:
                fig = self.plot_pv(self.sample_home, fig, data, file["name"], file)

            if 'battery' in type:
                fig = self.plot_battery(self.sample_home, fig, data, file["name"], file)

        return fig

    def plot_all_homes(self, fig=None):

        for file in (self.baselines + self.parametrics):
            with open(file["results"]) as f:
                data = json.load(f)

            fname = file["name"]
            for name, house in data.items():
                if name != "Summary":
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["temp_in_opt"], name=f"Tin (C) - {name} - {fname}"))
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["temp_wh_opt"], name=f"Twh (C) - {name} - {fname}"))
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["p_grid_opt"], name=f"Pgrid (kW) - {name} - {fname}", line_shape='hv', visible='legendonly'))
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["p_load_opt"], name=f"Pload (kW) - {name} - {fname}", line_shape='hv', visible='legendonly'))
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["hvac_cool_on_opt"], name=f"HVAC Cool Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["hvac_heat_on_opt"], name=f"HVAC Heat Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
                    fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=house["wh_heat_on_opt"], name=f"WH Heat Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)

        return fig

    def rl_simplified(self):
        flag = False

        for file in self.parametrics:
            with open(file['results']) as f:
                data = json.load(f)
            if flag == False:
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_setpoint"], name=f"Aggregate Load Setpoint"))
                setpoint = np.array(data["Summary"]["p_grid_setpoint"])
                flag = True
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_aggregate"], name=f"Aggregate Load - {file['name']}"))
            agg = np.array(data["Summary"]["p_grid_aggregate"])
            error = np.subtract(agg, 50*np.ones(len(agg)))
            # fig1.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(np.square(error)), name=f"L2 Norm Error {file['name']}"))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(abs(error)), name=f"Cummulative Error - {file['name']}"))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=abs(error), name=f"Abs Error - {file['name']}"))
            fig.update_layout(title_text="Aggregate Load")

        return fig

    def rl_simplified_rp(self, fig=None):
        for file in self.parametrics:
            with open(file['results']) as f:
                data = json.load(f)
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["RP"], name=f"Reward Price Signal - {file['name']}"))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(data["Summary"]["RP"]), np.arange(file['parent']['ts']) + 1), name=f"Rolling Average Reward Price - {file['name']}"))
            fig = self.plot_mu(fig)
            fig.update_layout(title_text="Reward Price Signal")
        return fig

    def plot_mu(self, fig):
        for file in self.parametrics:

            with open(file['results']) as f:
                data = json.load(f)

            try:
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["RP"], name=f"RP (Selected Action)", line_shape='hv'))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(data["Summary"]["RP"]), np.arange(file['parent']['ts']) + 1), name=f"Rolling Average Reward Price - {file['name']}"))
            except:
                self.ref_log.logger.warning("Could not find data on the selected action")
            with open(file['q_results']) as f:
                data = json.load(f)

            mus =[]
            for agent in data:
                agent_data = data[agent]
                mu = np.array(agent_data["mu"])
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=mu, name=f"Mu (Assumed Best Action) - {file['name']} - {agent}"))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=mu + file['params']['epsilon'], name=f"Mu +1 std dev - {file['name']} - {agent}", fill=None , mode='lines', line_color=plotly.colors.sequential.Blues[3]))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=mu - file['params']['epsilon'], name=f"Mu -1 std dev - {file['name']} - {agent}", fill='tonexty' , mode='lines', line_color=plotly.colors.sequential.Blues[3]))
                if len(mu) > 0:
                    mus.append(mu)
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.sum(mus, axis=0), name=f"Total Mu (RP without noise) - {file['name']}"))
        fig.update_layout(yaxis = {'exponentformat':'e'})
        fig.update_layout(title_text = "Reward Price Signal")
        return fig

    def plot_baseline(self, fig):
        for file in self.baselines:
            with open(file["results"]) as f:
                data = json.load(f)

            ts = len(data['Summary']['p_grid_aggregate'])-1
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_aggregate"], name=f"Agg Load - {file['name']}", line_shape='hv'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(np.divide(data["Summary"]["p_grid_aggregate"], file['parent']['agg_dt'])), name=f"Cumulative Agg Load - {file['name']}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(data["Summary"]["p_grid_aggregate"]), np.arange(ts) + 1), name=f"Avg Cumulative Agg Load - {file['name']}", line_shape='hv', visible='legendonly'))
        return fig

    def plot_parametric(self, fig):
        for file in self.parametrics:
            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]
            ts = len(data['Summary']['p_grid_aggregate'])-1
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_setpoint"], name=f"RL Setpoint Load - {name}"))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_aggregate"], name=f"Agg Load - RL - {name}", line_shape='hv'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(np.divide(data["Summary"]["p_grid_aggregate"],file['parent']['agg_dt'])), name=f"Cumulative Agg Load - RL - {name}", line_shape='hv', visible='legendonly'))
            # fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(data["Summary"]["p_grid_aggregate"][:ts+1]),np.arange(ts)+1), name=f"Avg Load - RL - {name}", line_shape='hv', visible='legendonly'))
            # fig = self.plot_mu(fig)
        return fig

    def plot_baseline_error(self, fig):
        for rl_file in self.parametrics:
            with open(rl_file['results']) as f:
                rldata = json.load(f)

            for file in self.baselines:
                with open(file['results']) as f:
                    data = json.load(f)

                rl2base_conversion = max(1, file['parent']['dt'] // rl_file['parent']['dt'])
                base2rl_conversion = max(1, rl_file['parent']['dt'] // file['parent']['dt'])
                base_load = np.repeat(data['Summary']['p_grid_aggregate'], base2rl_conversion)
                rl_setpoint = np.repeat(rldata['Summary']['p_grid_setpoint'], rl2base_conversion)
                rl_load = np.repeat(rldata['Summary']['p_grid_aggregate'], rl2base_conversion)
                if rl_setpoint[0] == 10:
                    rl_setpoint = rl_setpoint*3
                rl_setpoint = rl_setpoint[:len(rl_load)]
                rl_error = np.subtract(rl_load, rl_setpoint)
                base_error = np.subtract(base_load[:len(rl_setpoint)], rl_setpoint[:len(base_load)])
                rl2base_error = np.subtract(abs(base_error[:len(rl_setpoint)]), abs(rl_error[:len(base_load)]))/max(rl_file['parent']['dt'], file['parent']['dt'])

                if file['parent']['ts'] > rl_file['parent']['ts']:
                    x_lims = file['parent']['x_lims']
                    ts_max = file['parent']['ts']
                    dt = file['parent']['dt']
                else:
                    x_lims = rl_file['parent']['x_lims']
                    dt = rl_file['parent']['dt']
                    ts_max = rl_file['parent']['ts']

                ts = len(rl2base_error)
                fig.add_trace(go.Scatter(x=x_lims, y=rl2base_error, name=f"RL2Baseline Error - RL{rl_file['name']} and Baseline{file['name']}", visible='legendonly'))
                fig.add_trace(go.Scatter(x=x_lims, y=np.divide(np.cumsum(rl2base_error), (np.arange(ts)+1)), name=f"Avg RL2Baseline Error - RL{rl_file['name']} and Baseline{file['name']}", visible='legendonly'))

                fig.add_trace(go.Scatter(x=x_lims, y=base_error, name=f"Baseline Error - RL{rl_file['name']} and Baseline{file['name']}"))
                fig.add_trace(go.Scatter(x=x_lims, y=abs(base_error), name=f"Abs Baseline Error - RL{rl_file['name']} and Baseline{file['name']}"))

                hourly_base_error = np.zeros(np.int(np.ceil(len(base_error) / (24*ts_max)) * (24*ts_max)))
                hourly_base_error[:len(base_error)] = abs(base_error)
                hourly_base_error = hourly_base_error.reshape(dt,-1).sum(axis=0)
                fig.add_trace(go.Scatter(x=x_lims[::dt], y=hourly_base_error, name=f"Baseline Hourly Error - RL{rl_file['name']} and Baseline{file['name']}"))
                fig.add_trace(go.Scatter(x=x_lims[::dt], y=np.cumsum(hourly_base_error), name=f"Cumulative Baseline Hourly Error - RL{rl_file['name']} and Baseline{file['name']}"))

                period = self.config['simulation']['checkpoint_interval']
                period_hours = {"hourly": 1, "daily": 24, "weekly": 7*24}
                if not period in period_hours:
                    if isinstance(period, int):
                        period_hours[period] = period
                num_periods = int(np.ceil(len(hourly_base_error) / period_hours[period]))
                periodic_acum_error = np.zeros(num_periods * period_hours[period])
                periodic_acum_error[:len(hourly_base_error)] = hourly_base_error
                periodic_acum_error = hourly_base_error.reshape(num_periods, -1)
                periodic_acum_error = np.cumsum(periodic_acum_error, axis=1).flatten()
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'][::max(rl_file['parent']['dt'], file['parent']['dt'])], y=periodic_acum_error, name=f"Accumulated Baseline Hourly Error - RL{rl_file['name']} and Baseline{file['name']}", visible='legendonly'))
        return fig

    def plot_parametric_error(self, fig):
        for file in self.parametrics:
            with open(file['results']) as f:
                data = json.load(f)

            name = file['name']
            rl_load = data['Summary']['p_grid_aggregate']
            rl_setpoint = data['Summary']['p_grid_setpoint']
            rl_error = np.subtract(rl_load[:len(rl_setpoint)], rl_setpoint[:len(rl_load)]) / file['parent']['agg_dt']
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=rl_error, name=f"Error - {name} (kWh)", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(rl_error), np.arange(len(rl_error))+1), name=f"Average Error - {name} (kWh)", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=rl_setpoint, name=f"Setpoint - {name} (kW)", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=abs(rl_error), name=f"Abs Error - {name} (kWh)", line_shape='hv', visible='legendonly'))

            hourly_rl_error = np.zeros(np.int(np.ceil(len(rl_error) / (24*file['parent']['agg_dt'])) * (24*file['parent']['agg_dt'])))
            hourly_rl_error[:len(rl_error)] = abs(rl_error)
            hourly_rl_error = hourly_rl_error.reshape(file['parent']['agg_dt'],-1).sum(axis=0)
            hourly_rl_error = np.repeat(hourly_rl_error, file['parent']['agg_dt'])
            print("standard deviation of p_grid", file['name'], np.std(rl_load))
            print("standard deviation of p_grid excluding first day", file['name'], np.std(rl_load[24:]))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=hourly_rl_error, name=f"Hourly Error - {name} (kWh)", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(hourly_rl_error/file['parent']['agg_dt']), name=f"Cumulative Hourly Error - {name} (kWh)", line_shape='hv', visible='legendonly'))

            period = self.config['simulation']['checkpoint_interval']
            period_hours = {"hourly": 1, "daily": 24, "weekly": 7*24}
            if not period in period_hours:
                if isinstance(period, int):
                    period_hours[period] = period
            num_periods = int(np.ceil(len(hourly_rl_error) / (period_hours[period]*file['parent']['agg_dt'])))
            periodic_acum_error = np.zeros(num_periods * period_hours[period])
            periodic_acum_error[:len(hourly_rl_error)] = hourly_rl_error[::file['parent']['agg_dt']]
            periodic_acum_error = hourly_rl_error.reshape(num_periods, -1)
            periodic_acum_error = np.cumsum(periodic_acum_error, axis=1).flatten()
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=periodic_acum_error/file['parent']['agg_dt'], name=f"Accumulated Hourly Error - {name}", visible='legendonly'))
        return fig

    def plot_rewards(self, fig):
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            try:
                data = data["horizon"]
            except:
                data = data["next"]
            name = file["name"]
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["average_reward"], name=f"Average Reward - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["cumulative_reward"], name=f"Cumulative Reward - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["reward"], name=f"Reward - {name}", line_shape='hv', visible='legendonly'), secondary_y=True)
        return fig

    def just_the_baseline(self, fig):
        if len(self.baselines) == 0:
            self.ref_log.logger.error("No baseline run files found for analysis.")
        fig = self.plot_baseline(fig)
        fig.update_layout(title_text="Baseline Summary")
        return fig

    def rl2baseline(self, fig):
        if len(self.parametrics) == 0:
            self.ref_log.logger.warning("No parameterized RL aggregator runs found for comparison to baseline.")
            fig = self.just_the_baseline(fig)
            return fig
        # fig = self.plot_greedy(fig)
        fig = self.plot_baseline(fig)
        fig = self.plot_parametric(fig)
        fig.update_layout(title_text="RL Baseline Comparison")
        return fig

    def rl2baseline_error(self, fig):
        fig = self.plot_baseline_error(fig)
        fig = self.plot_parametric_error(fig)
        # fig = self.plot_rewards(fig)
        fig.update_layout(title_text="RL Baseline Error Metrics")
        return fig

    def q_values(self, fig):
        with open(rl_q_file) as f:
            data = json.load(f)

        x1 = []
        x2 = []
        for i in data["state"]:
            if i[0] < 0:
                x1.append(i[0])
            else:
                x2.append(i[0])
        fig.add_trace(go.Scatter3d(x=x1, y=data["action"], z=data["q_obs"], mode="markers"))
        fig.add_trace(go.Scatter3d(x=x2, y=data["action"], z=data["q_obs"], mode="markers"))
        return fig

    def rl_qvals(self, fig):
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["q_pred"], name=f"Q predicted - {file['name']}", marker={'opacity':0.2}))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["q_obs"], name=f"Q observed - {file['name']}"))

        fig.update_layout(title_text="Critic Network")
        fig.show()
        return fig

    def rl_thetas(self, fig):
        counter = 1
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            data = data["horizon"]
            theta = data["theta"]

            for i in range(len(data["theta"][0])):
                y = []
                for j in range(file['parent']['ts']):
                    y.append(theta[j][i])
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=y, name=f"Theta_{i}", line_shape='hv', legendgroup=file['name']))
            counter += 1
        fig.update_layout(title_text="Critic Network Coefficients")
        return fig

    def all_rps(self, fig):
        for file in self.parametrics:
            with open(file['results']) as f:
                data = json.load(f)

            rps = data['Summary']['RP']
            fig.add_trace(go.Histogram(x=rps, name=f"{file['name']}"), row=1, col=1)

            with open(file['q_results']) as f:
                data = json.load(f)
            data = data["horizon"]
            mu = np.array(data["mu"])
            std = self.config['rl']['parameters']['exploration_rate'][0]
            delta = np.subtract(mu, rps)

            fig.add_trace(go.Histogram(x=delta, name=f"{file['name']}"), row=2, col=1)
            fig.add_trace(go.Scatter(x=[-std, -std, std, std], y=[0, 0.3*len(rps), 0.3*len(rps), 0], fill="toself"), row=2, col=1)
        return fig

if __name__ == "__main__":
    r = Reformat()
    r.main()
