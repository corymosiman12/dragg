import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import itertools as it
import random

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dragg.logger import Logger
import dragg.aggregator as agg

class Reformat:
    def __init__(self, outputs_dir={'outputs'}, agg_params={}, mpc_params={}, date_ranges={}, include_runs={}, log=Logger("reformat")):
        self.ref_log = log
        self.data_dir = 'data'
        self.outputs_dir = outputs_dir
        if not os.path.isdir(self.outputs_dir):
            self.ref_log.error("Outputs directory does not exist.")
            quit()
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.json'))
        self.config = self._import_config()
        self.add_date_ranges = date_ranges
        self.date_ranges = self._setup_date_ranges()
        self.date_folders = self.set_date_folders()

        self.add_mpc_params = mpc_params
        self.mpc_params = self._setup_mpc_params()
        self.mpc_folders = self.set_mpc_folders()
        self.baselines = []
        self.set_base_file()

        self.add_agg_params = agg_params
        self.agg_params = self._setup_agg_params()
        self.include_runs = include_runs
        self.parametrics = []
        self.set_parametric_files()

        np.random.seed(self.config["random_seed"])

    def _setup_date_ranges(self):
        start_dates = [datetime.strptime(self.config["start_datetime"], '%Y-%m-%d %H')]
        end_dates = set([datetime.strptime(self.config["end_datetime"], '%Y-%m-%d %H')])
        temp = {"start_datetime": start_dates, "end_datetime": end_dates}
        for key in temp:
            if key in self.add_date_ranges:
                temp[key].add(datetime.strptime(self.add_date_ranges[key], '%Y-%m-%d %H'))
        return temp

    def _setup_agg_params(self):
        alphas = set(self.config["rl_learning_rate"])
        epsilons = set(self.config["rl_exploration_rate"])
        betas = set(self.config["rl_discount_factor"])
        batch_sizes = set(self.config["rl_batch_size"])
        rl_horizons = set(self.config["rl_agg_action_horizon"])
        mpc_disutil = set(self.config["mpc_disutility"])
        mpc_discomf = set(self.config["mpc_discomfort"])
        temp = {"alpha": alphas, "epsilon": epsilons, "beta": betas, "batch_size": batch_sizes, "rl_horizon": rl_horizons, "mpc_disutility": mpc_disutil, "mpc_discomfort": mpc_discomf}
        for key in temp:
            if key in self.add_agg_params:
                temp[key] |= set(self.add_agg_params[key])
        return temp

    def _setup_mpc_params(self):
        n_houses = self.config["total_number_homes"]
        mpc_horizon = self.config["mpc_prediction_horizons"]
        dt = self.config["mpc_hourly_steps"]
        check_type = self.config["check_type"]
        mpc_discomf = set(self.config["mpc_discomfort"])
        temp = {"n_houses": set([n_houses]), "mpc_horizon": set(mpc_horizon), "mpc_hourly_steps": set([dt]), "check_type": set([check_type]), "mpc_discomfort": mpc_discomf}
        for key in temp:
            if key in self.add_mpc_params:
                temp[key] |= set(self.add_mpc_params[key])
        return temp

    def set_date_folders(self):
        temp = []
        keys, values = zip(*self.date_ranges.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        permutations = sorted(permutations, key=lambda i: i['end_datetime'], reverse=True)
        for j in self.outputs_dir:
            for i in permutations:
                date_folder = os.path.join(j, f"{i['start_datetime'].strftime('%Y-%m-%dT%H')}_{i['end_datetime'].strftime('%Y-%m-%dT%H')}")
                if os.path.isdir(date_folder):
                    hours = i['end_datetime'] - i['start_datetime']
                    hours = int(hours.total_seconds() / 3600)
                    new_folder = {"folder": date_folder, "hours": hours, "start_dt": i['start_datetime'], "name": j}
                    temp.append(new_folder)
        if len(temp) == 0:
            self.ref_log.logger.error("No files found for the date ranges specified.")
            exit()
        return temp

    def set_mpc_folders(self):
        temp = []
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        for j in self.date_folders:
            for i in permutations:
                interval_minutes = 60 // i['mpc_hourly_steps']
                mpc_folder = os.path.join(j["folder"], f"{i['check_type']}-homes_{i['n_houses']}-horizon_{i['mpc_horizon']}-interval_{interval_minutes}")
                if os.path.isdir(mpc_folder):
                    timesteps = j['hours']*i['mpc_hourly_steps']
                    x_lims = [j['start_dt'] + timedelta(minutes=x*interval_minutes) for x in range(timesteps + max(self.config["rl_agg_action_horizon"])*i['mpc_hourly_steps'])]
                    name = j['name']
                    for k,v in i.items():
                        if len(self.mpc_params[k]) > 1:
                            name += f"{k} = {v}, "
                    set = {'path': mpc_folder, 'dt': i['mpc_hourly_steps'], 'ts': timesteps, 'x_lims': x_lims, 'name': name}
                    if not mpc_folder in temp:
                        temp.append(set)
        return temp

    def set_base_file(self):
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        for j in self.mpc_folders:
            path = j['path']
            for i in permutations:
                file = os.path.join(path, "baseline", f"baseline_discomf-{float(i['mpc_discomfort'])}-results.json")
                if os.path.isfile(file):
                    name = f"Baseline - {j['name']}"
                    temp = {"results": file, "name": name, "parent": j}
                    self.baselines.append(temp)

    def set_rl_files(self):
        counter = 1
        for i in self.mpc_folders:
            path = i['path']
            rl_agg_folder = os.path.join(path, "rl_agg")
            keys, values = zip(*self.agg_params.items())
            permutations = [dict(zip(keys, v)) for v in it.product(*values)]
            for j in permutations:
                if os.path.isdir(rl_agg_folder):
                    rl_agg_path = f"agg_horizon_{j['rl_horizon']}-alpha_{j['alpha']}-epsilon_{j['epsilon']}-beta_{j['beta']}_batch-{j['batch_size']}_disutil-{float(j['mpc_disutility'])}_discomf-{float(j['mpc_discomfort'])}"
                    results_file = rl_agg_path + "-results.json"
                    rl_agg_file = os.path.join(rl_agg_folder, results_file)
                    if os.path.isfile(rl_agg_file):
                        q_results = rl_agg_path + "-q-results.json"
                        q_file = os.path.join(rl_agg_folder, q_results)
                        name = i["name"]
                        for k,v in j.items():
                            if len(self.agg_params[k]) > 1:
                                name += f"{k} = {v}, "
                        if name=="":
                            name += f"RL {counter}"
                            counter += 1
                        # name =  f"horizon={j['rl_horizon']}, alpha={j['alpha']}, beta={j['beta']}, epsilon={j['epsilon']}, batch={j['batch_size']}, disutil={j['mpc_disutility']}, discomf={j['mpc_discomfort']}"
                        set = {"results": rl_agg_file, "q_results": q_file, "name": name, "parent": i, "rl_agg_action_horizon": j["rl_horizon"]}
                        self.parametrics.append(set)
        if len(self.parametrics) == 0:
            self.ref_log.logger.warning("Parameterized RL aggregator runs are empty for this config file.")

    def set_simplified_files(self):
        for i in self.mpc_folders:
            path = i['path']
            simplified_folder = os.path.join(path, "simplified")
            keys, values = zip(*self.agg_params.items())
            permutations = [dict(zip(keys, v)) for v in it.product(*values)]
            for j in permutations:
                if os.path.isdir(simplified_folder):
                    simplified_path = f"agg_horizon_{j['rl_horizon']}-alpha_{j['alpha']}-epsilon_{j['epsilon']}-beta_{j['beta']}_batch-{j['batch_size']}"
                    results_file = simplified_path + "-results.json"
                    simplified_file = os.path.join(simplified_folder, results_file)
                    if os.path.isfile(simplified_file):
                        q_results = simplified_path + "-q-results.json"
                        q_file = os.path.join(simplified_folder, q_results)
                        name = f"Simplified Response - horizon={j['rl_horizon']}, alpha={j['alpha']}, beta={j['beta']}, epsilon={j['epsilon']}, batch={j['batch_size']}"
                        set = {"results": simplified_file, "q_results": q_file, "name": name, "parent": i}
                        self.parametrics.append(set)

    def set_parametric_files(self):
        if self.config["run_rl_agg"] or "rl_agg" in self.include_runs:
            self.set_rl_files()
        if self.config["run_simplified"] or "simplified" in self.include_runs:
            self.set_simplified_files()

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
            data = json.load(f)
        return data

    def _config_summary(self):
        s = []
        for d in self.data:
            s.append(d["Summary"])
        return s

    def plot_baseline_summary(self):
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["p_grid_aggregate"], name="Pagg (kW)"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["SPP"], name="Energy Cost ($/kWh)"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["OAT"], name="OAT (C)"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["GHI"], name="GHI (W/m2)"),
                      row=1, col=2)
        #
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_baseline_summary2(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["p_grid_aggregate"], name="Pagg (kW)"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["OAT"], name="OAT (C)"), secondary_y=True)
        # fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["SPP"], name="Energy Cost ($/kWh)"))
        # fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["GHI"], name="GHI (W/m2"))

        fig.update_yaxes(title_text='Power (kW)', secondary_y=False)
        fig.update_yaxes(title_text='Temperature (C)', secondary_y=True)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_agg_vs_homes(self, names):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=self.summary["p_grid_aggregate"], name="Pagg (kW)"))
        for home, data in self.baseline_data.items():
            # if home in names:
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_grid_opt"], opacity=0.25, name=f"{home}-{data['type']}"), secondary_y=True)

        fig.update_yaxes(title_text='Pagg (kW)', secondary_y=False)
        fig.update_yaxes(title_text='Pload (kW)', secondary_y=True)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_environmental_values(self, fig, summary, file):
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=summary["OAT"][0:file["parent"]["ts"]], name=f"OAT (C)"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=summary["GHI"][0:file["parent"]["ts"]], name=f"GHI (W/m2)"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=summary["TOU"][0:file["parent"]["ts"]], name=f"TOU Price ($/kWh)", line_shape='hv'), secondary_y=True)
        return fig

    def plot_single_home_base(self, name, fig, data, summary, fname, file):
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["temp_in_opt"], name=f"Tin (C) - {fname}"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["temp_wh_opt"], name=f"Twh (C) - {fname}"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["temp_in_sp"] * np.ones(file["parent"]["ts"]), name=f"Tin_sp (C) - {fname}"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["temp_wh_sp"] * np.ones(file["parent"]["ts"]), name=f"Twh_sp (C) - {fname}"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_grid_opt"], name=f"Pgrid (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_load_opt"], name=f"Pload (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["hvac_cool_on_opt"], name=f"HVAC Cool Cmd - {fname}", line_shape='hv'), secondary_y=True)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["hvac_heat_on_opt"], name=f"HVAC Heat Cmd - {fname}", line_shape='hv'), secondary_y=True)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["wh_heat_on_opt"], name=f"WH Heat Cmd - {fname}", line_shape='hv'), secondary_y=True)
        try: # only for aggregator files
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.add(summary["TOU"], summary["RP"]), name=f"Actual Price ($/kWh) - {fname}", line_shape='hv'), secondary_y=True)
        except:
            pass
        return fig

    def plot_single_home_pv(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_pv_opt"], name=f"Ppv (kW) - {fname}", line_shape='hv'))
        return fig

    def plot_single_home_battery(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["e_batt_opt"], name=f"SOC (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_batt_ch"], name=f"Pch (kW) - {fname}", line_shape='hv'))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["p_batt_disch"], name=f"Pdis (kW) - {fname}", line_shape='hv'))
        return fig

    def plot_single_home2(self, name=None, type=None):
        if name is None:
            if type is None:
                type = "base"
                self.ref_log.logger.warning("Specify a home type or name. Proceeding with home of type: \"base\".")

            type_list = self._type_list(type)
            name = random.sample(type_list,1)[0]
            self.ref_log.logger.info(f"Proceeding with home: {name}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        flag = False
        for file in (self.baselines + self.parametrics):
            with open(file["results"]) as f:
                comm_data = json.load(f)

            try:
                data = comm_data[name]
            except:
                self.ref_log.logger.error(f"No home with name: {name}")
                return

            type = data["type"]
            summary = comm_data["Summary"]
            horizon = summary["horizon"]

            if not flag:
                fig = self.plot_environmental_values(fig, summary, file)
                flag = True

            fig = self.plot_single_home_base(name, fig, data, summary, file["name"], file)

            case = summary["case"]
            fig.update_xaxes(title_text="Time of Day (hour)")
            fig.update_layout(title_text=f"{name} - {type} type")

            if 'pv' in type:
                fig = self.plot_single_home_pv(name, fig, data, file["name"], file)

            if 'battery' in type:
                fig = self.plot_single_home_battery(name, fig, data, file["name"], file)

        fig.show()

    def plot_all_homes(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for file in (self.baselines + self.parametrics):
            with open(file["results"]) as f:
                data = json.load(f)

            fname = file["name"]
            for name, house in data.items():
                if name != "Summary":
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["temp_in_opt"], name=f"Tin (C) - {name} - {fname}"))
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["temp_wh_opt"], name=f"Twh (C) - {name} - {fname}"))
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["p_grid_opt"], name=f"Pgrid (kW) - {name} - {fname}", line_shape='hv', visible='legendonly'))
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["p_load_opt"], name=f"Pload (kW) - {name} - {fname}", line_shape='hv', visible='legendonly'))
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["hvac_cool_on_opt"], name=f"HVAC Cool Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["hvac_heat_on_opt"], name=f"HVAC Heat Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)
                    fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=house["wh_heat_on_opt"], name=f"WH Heat Cmd - {name} - {fname}", line_shape='hv', visible='legendonly'), secondary_y=True)

        fig.show()

    def compare_agg_between_runs(self, n_homes):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for n, s in enumerate(self.summary_data):
            if n == 0:
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=s["SPP"], name=f"TOU Price"), secondary_y=True)
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=s["p_grid_aggregate"], name=f"Pagg H: {s['horizon']}"))

        fig.update_yaxes(title_text="Power (kW)")
        fig.update_yaxes(title_text="Cost ($/kWh)", secondary_y=True)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {n_homes} - {self.check_type} type")
        fig.show()

    def calc_agg_costs(self):
        self.agg_costs = []
        for run in self.data:
            all = 0
            base_cost = 0
            pv_cost = 0
            battery_cost = 0
            pv_battery_cost = 0
            for k, v in run.items():
                if k != "Summary":
                    if v["type"] == "base":
                        base_cost += sum(v["cost_opt"])
                    elif v["type"] == "pv_only":
                        pv_cost += sum(v["cost_opt"])
                    elif v["type"] == "battery_only":
                        battery_cost += sum(v["cost_opt"])
                    elif v["type"] == "pv_battery":
                        pv_battery_cost += sum(v["cost_opt"])
            all = base_cost + pv_cost + pv_battery_cost + battery_cost
            self.agg_costs.append({
                "all": all,
                "base_cost": base_cost,
                "pv_cost": pv_cost,
                "battery_cost": battery_cost,
                "pv_battery_cost": pv_battery_cost
            })

    def computation_time_vs_horizon_vs_agg_cost(self, n_homes):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = [x['horizon'] for x in self.summary_data]
        comp_time = [x["solve_time"] for x in self.summary_data]
        agg_cost = [x["all"] for x in self.agg_costs]

        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg_cost, name="Aggregate Cost"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=comp_time, name="Solve Time"), secondary_y=True)

        fig.update_yaxes(title_text="Cost ($)")
        fig.update_yaxes(title_text="Solve Time (seconds)", secondary_y=True)
        fig.update_xaxes(title_text="Prediction Horizon")
        fig.update_layout(title_text=f"Baseline - {n_homes} homes")
        fig.show()

    def plot_greedy(self, fig, file):
        with open(file['q_results']) as f:
            data = json.load(f)

        rtgs = {}
        for i in range(file["rl_agg_action_horizon"]+1):
            rtgs[i] = []

        num_init_steps = (file["rl_agg_action_horizon"]*file['parent']['dt'])
        is_greedy = [False]*num_init_steps
        is_greedy += data['is_greedy']
        is_random = [int(not(i)) for i in is_greedy]
        is_greedy = [int(i) for i in is_greedy[num_init_steps:]]

        fig.add_trace(go.Bar(x=file["parent"]["x_lims"], y=is_random, name="Random Action - Current", marker={'color':'purple', 'opacity':0.3}), secondary_y=True)
        num_ts = file["parent"]["ts"]
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.divide(np.cumsum(np.multiply(is_greedy, data["reward"])), np.cumsum(is_greedy) + 1), name=f"Greedy Action Average Reward - {file['name']}"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.divide(np.cumsum(data["reward"]), np.arange(num_ts) + 1), name=f"Numerical avg. reward - {file['name']}"))
        # fig.add_trace(go.Bar(name="Random Action - Forecast", x=file["parent"]["x_lims"], y=, marker={'color':'orange', 'opacity':0.3}), secondary_y=True)

        return fig

    def plot_baseline(self, fig):
        for file in self.baselines:
            with open(file["results"]) as f:
                data = json.load(f)

            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["Summary"]["p_grid_aggregate"], name=f"Agg Load - {file['name']}", visible='legendonly'))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.cumsum(data["Summary"]["p_grid_aggregate"]), name=f"Cumulative Agg Load - {file['name']}", visible='legendonly'))
        return fig

    def plot_parametric(self, fig):
        if self.parametrics[0]:
            file = self.parametrics[0]
            with open(file["results"]) as f:
                rldata = json.load(f)

            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rldata["Summary"]["p_grid_setpoint"], name="RL Setpoint Load"))

        for file in self.parametrics:
            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["Summary"]["p_grid_aggregate"][1:], name=f"Agg Load - RL - {name}"))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.cumsum(data["Summary"]["p_grid_aggregate"][1:]), name=f"Cumulative Agg Load - RL - {name}", visible='legendonly'))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.divide(np.cumsum(data["Summary"]["p_grid_aggregate"][1:file["parent"]["ts"]+1]),np.arange(file["parent"]["ts"])+1), name=f"Avg Load - RL - {name}", visible='legendonly'))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["Summary"]["RP"], name=f"RP - RL - {name}", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.divide(np.cumsum(data["Summary"]["RP"])[:file["parent"]["ts"]], np.arange(file["parent"]["ts"]) + 1), name=f"Average RP", visible='legendonly'), secondary_y=True)
            self.plot_greedy(fig, file)
            # fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=np.add(data["Summary"]["TOU"], data["Summary"]["RP"]).tolist(), name="Actual Price ($/kWh)", line_shape='hv'), secondary_y=True)

        return fig

    def plot_baseline_error(self, fig):
        if self.parametrics[0]:
            max_file = self.parametrics[np.argmax([file['parent']['ts'] for file in self.parametrics])]
            max_ts = max_file['parent']['ts']
            max_x_lims = max_file['parent']['x_lims']

            with open(max_file["results"]) as f:
                rldata = json.load(f)

            fig.add_trace(go.Scatter(x=max_x_lims, y=rldata["Summary"]["p_grid_setpoint"], name="RL Setpoint Load"))

        for file in self.baselines:
            with open(file["results"]) as f:
                data = json.load(f)

            scale = max_ts // file['parent']['ts']
            baseline_load = np.repeat(data["Summary"]["p_grid_aggregate"], scale)
            baseline_setpoint = rldata["Summary"]["p_grid_setpoint"]
            baseline_error = np.subtract(baseline_load, baseline_setpoint)
            fig.add_trace(go.Scatter(x=max_x_lims, y=(baseline_error), name=f"Error - {file['name']}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=abs(baseline_error), name=f"Abs Error - {file['name']}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(np.abs(baseline_error)), name=f"Abs Cummulative Error - {file['name']}", line_shape='hv'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(baseline_error), name=f"Cummulative Error - {file['name']}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.divide(np.cumsum(baseline_error)[:file["parent"]["ts"]],np.arange(file["parent"]["ts"]) + 1), name=f"Average Error - {file['name']}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(np.square(baseline_error)), name=f"L2 Norm Cummulative Error - {file['name']}", line_shape='hv'))

        return fig

    def plot_parametric_error(self, fig):
        max_file = self.parametrics[np.argmax([file['parent']['ts'] for file in self.parametrics])]
        max_ts = max_file['parent']['ts']
        max_x_lims = max_file['parent']['x_lims']
        for file in self.parametrics:
            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]
            rl_load = data["Summary"]["p_grid_aggregate"][1:]
            rl_setpoint = data["Summary"]["p_grid_setpoint"]
            scale = max_ts // file['parent']['ts']
            rl_load = np.repeat(rl_load, scale)
            rl_setpoint = np.repeat(rl_setpoint, scale)
            rl_error = np.subtract(rl_load, rl_setpoint)
            fig.add_trace(go.Scatter(x=max_x_lims, y=(rl_error), name=f"Error - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=abs(rl_error), name=f"Abs Error - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(np.abs(rl_error)), name=f"Cummulative Abs Error - {name}", line_shape='hv'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(rl_error), name=f"Cummulative Error - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.divide(np.cumsum(rl_error),np.arange(max_ts)+1), name=f"Average Error - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=max_x_lims, y=np.cumsum(np.square(rl_error)), name=f"L2 Norm Cummulative Error - {name}", line_shape='hv'))

        return fig

    def plot_rewards(self, fig):
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            name = file["name"]
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["average_reward"], name=f"Average Reward - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["cumulative_reward"], name=f"Cumulative Reward - {name}", line_shape='hv', visible='legendonly'))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["reward"], name=f"Reward - {name}", line_shape='hv'))
            try:
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["best_action"], name="Best Action", line_shape='hv', visible='legendonly'))
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["second"], name="second Action", line_shape='hv', visible='legendonly'))
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["third"], name="third Action", line_shape='hv', visible='legendonly'))
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["fourth"], name="fourth Action", line_shape='hv', visible='legendonly'))
            except:
                pass
            self.plot_greedy(fig, file)
        return fig

    def rl2baseline(self):
        if len(self.parametrics) == 0:
            self.ref_log.logger.error("No parameterized RL aggregator runs found for comparison to baseline.")
            return

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # fig = self.plot_greedy(fig)
        fig = self.plot_baseline(fig)
        fig = self.plot_parametric(fig)

        fig.show()

    def rl2baseline_error(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig = self.plot_baseline_error(fig)
        fig = self.plot_parametric_error(fig)
        fig = self.plot_rewards(fig)
        fig.update_layout(title_text="RL and Baseline Error")

        fig.show()

    def rl_qtables(self, rl_q_file):
        with open(rl_q_file) as f:
            data = json.load(f)

        fig = make_subplots()
        x = []
        y = []
        t = 100
        for i in range(len(data["q_tables"][t])):
            x.append(data[t][i][0])
            y.append(data[t][i][1])

        fig.add_trace(go.Scatter(x=x, y=y))

        fig.show()

    def q_values(self, rl_q_file):
        with open(rl_q_file) as f:
            data = json.load(f)

        x1 = []
        x2 = []
        for i in data["state"]:
            if i[0] < 0:
                x1.append(i[0])
            else:
                x2.append(i[0])
        fig = make_subplots()
        fig.add_trace(go.Scatter3d(x=x1, y=data["action"], z=data["q_obs"], mode="markers"))
        fig.add_trace(go.Scatter3d(x=x2, y=data["action"], z=data["q_obs"], mode="markers"))
        fig.show()

    def rl_qvals(self):
        fig = make_subplots()
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["q_obs"], name=f"Q observed - {file['name']}"))
            fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=data["q_pred"], name=f"Q predicted - {file['name']}"))

        fig.show()

    def rl_thetas(self):
        fig = make_subplots(rows=2, cols=len(self.parametrics))
        counter = 1
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

            theta = data["theta"]
            phi = data["phi"]

            # x = np.arange(self.hours)
            for i in range(len(data["theta"][0])):
                y = []
                z = []
                for j in range(file['parent']['ts']):
                    y.append(theta[j][i])
                    z.append(phi[j][i])
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=y, name=f"Theta_{i} - {file['name']}", line_shape='hv', legendgroup=file['name']),1,counter)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=z, name=f"Phi_{i} - {file['name']}", line_shape='hv'),2,counter)
            counter += 1

        fig.show()

    def rl_phis(self):
        fig = make_subplots()
        for file in self.parametrics:
            with open(file["q_results"]) as f:
                data = json.load(f)

        phi = data["phi"]


    def reward_prices_over_time(self):
        file = os.path.join(self.outputs_dir, "agg_mpc", "2015-01-01T00_2015-01-02T00-agg_mpc_all-homes_20-horizon_8-iter-results.json")
        with open(file) as f:
            data = json.load(f)

        fig = make_subplots(rows=3, cols=1)
        rp = []
        for timestep in data:
            x = [x for x in range(len(timestep['agg_cost']))]
            rp.append(timestep['reward_price'][-1])
            if len(x) > 1:
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=timestep['agg_cost'], name=f"TS: {timestep['timestep']}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=timestep['agg_load'], name=f"TS: {timestep['timestep']}"), row=2, col=1)

        x2 = [x for x in range(24)]
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rp, name=f"Reward Price"), row=3, col=1)

        fig.update_yaxes(title_text='Cost ($)', row=1, col=1)
        fig.update_yaxes(title_text='Load (kW)', row=2, col=1)
        fig.update_yaxes(title_text='Reward ($)', row=3, col=1)
        fig.update_xaxes(title_text="Aggregator Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Aggregator Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Timestep", row=3, col=1)
        fig.update_layout(title_text=f"Aggregate Costs and Loads at Different Timesteps")
        fig.show()

    def single_home_rbo_mpc_vs_agg_mpc(self, name):
        rbo_mpc_file = os.path.join(self.outputs_dir, "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_8-results.json")
        agg_mpc_file = os.path.join(self.outputs_dir, "2015-01-01T00_2015-01-02T00-agg_mpc_all-homes_20-horizon_8-results.json")

        with open(rbo_mpc_file) as f:
            rbo = json.load(f)

        with open(agg_mpc_file) as f:
            agg = json.load(f)
        rbo = rbo[name]
        agg = agg[name]
        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["temp_in_opt"][0:self.hours], name="RBO MPC Tin"), row=1, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["temp_in_opt"][0:self.hours], name="AGG MPC Tin"), row=1, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["hvac_heat_on_opt"][0:self.hours], name="RBO MPC HVAC Heat CMD"), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["hvac_heat_on_opt"][0:self.hours], name="AGG MPC HVAC Heat CMD"), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["temp_wh_opt"][0:self.hours], name="RBO MPC Twh"), row=2, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["temp_wh_opt"][0:self.hours], name="AGG MPC Twh"), row=2, col=1)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["wh_heat_on_opt"][0:self.hours], name="RBO MPC WH Heat CMD"), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["wh_heat_on_opt"][0:self.hours], name="AGG MPC WH Heat CMD"), row=2, col=1, secondary_y=True)
        fig.update_layout(title_text=f"{name}")
        fig.show()

    def single_home_rbo_mpc_vs_agg_mpc2(self):
        rbo_mpc_file = os.path.join(self.outputs_dir,
                                    "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_8-results.json")
        agg_mpc_file = os.path.join(self.outputs_dir,
                                    "2015-01-01T00_2015-01-02T00-agg_mpc_all-homes_20-horizon_8-results.json")

        with open(rbo_mpc_file) as f:
            rbo2 = json.load(f)

        with open(agg_mpc_file) as f:
            agg2 = json.load(f)
        for k, v in rbo2.items():
            if k != "Summary":
                rbo = rbo2[k]
                agg = agg2[k]
                fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True}],
                                                           [{"secondary_y": True}],
                                                           [{"secondary_y": True}],
                                                           [{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["temp_in_opt"][0:self.hours], name="RBO MPC Tin"), row=1,
                              col=1)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["temp_in_opt"][0:self.hours], name="AGG MPC Tin"), row=1,
                              col=1)
                fig.add_trace(
                    go.Scatter(x=file["parent"]["x_lims"], y=rbo["hvac_heat_on_opt"][0:self.hours], name="RBO MPC HVAC Heat CMD"), row=1,
                    col=1, secondary_y=True)
                fig.add_trace(
                    go.Scatter(x=file["parent"]["x_lims"], y=agg["hvac_heat_on_opt"][0:self.hours], name="AGG MPC HVAC Heat CMD"), row=1,
                    col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["temp_wh_opt"][0:self.hours], name="RBO MPC Twh"), row=2,
                              col=1)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["temp_wh_opt"][0:self.hours], name="AGG MPC Twh"), row=2,
                              col=1)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo["wh_heat_on_opt"][0:self.hours], name="RBO MPC WH Heat CMD"),
                              row=2, col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg["wh_heat_on_opt"][0:self.hours], name="AGG MPC WH Heat CMD"),
                              row=2, col=1, secondary_y=True)
                fig.add_trace(
                    go.Scatter(x=file["parent"]["x_lims"], y=rbo["p_grid_opt"][0:self.hours], name="RBO MPC Pgrid"),
                    row=3, col=1)
                fig.add_trace(
                    go.Scatter(x=file["parent"]["x_lims"], y=agg["p_grid_opt"][0:self.hours], name="AGG MPC Pgrid"),
                    row=3, col=1)
                if 'battery' in rbo['type']:
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=rbo["p_batt_disch"][0:self.hours], name="RBO MPC Pdis"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=agg["p_batt_disch"][0:self.hours], name="AGG MPC Pdis"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=rbo["p_batt_ch"][0:self.hours], name="RBO MPC Pch"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=agg["p_batt_ch"][0:self.hours], name="AGG MPC Pch"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=rbo["e_batt_opt"][0:self.hours], name="RBO MPC Ebatt"),
                        row=4, col=1)
                    fig.add_trace(
                        go.Scatter(x=file["parent"]["x_lims"], y=agg["e_batt_opt"][0:self.hours], name="AGG MPC Ebatt"),
                        row=4, col=1)

                fig.update_yaxes(title_text="CMD", row=1, col=1, secondary_y=True, range=[0, 4])
                fig.update_yaxes(title_text="CMD", row=2, col=1, secondary_y=True, range=[0, 4])
                fig.update_yaxes(title_text="Temp (C)", row=1, col=1, range=[18, 21], secondary_y=False)
                fig.update_yaxes(title_text="Temp (C)", row=2, col=1, range=[44, 48], secondary_y=False)
                fig.update_yaxes(title_text="Power (kW)", row=3, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Ebatt (kWh)", row=4, col=1)
                fig.update_layout(title_text=f"{k}-{rbo['type']}")
                fig.show()

    def baseline_vs(self):
        baseline = self.data[0]["Summary"]
        rbo_mpc = self.data[3]["Summary"]
        agg_mpc = self.data[-1]["Summary"]

        fig = make_subplots()
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=baseline["p_grid_aggregate"], name=f"Pagg Baseline"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=rbo_mpc["p_grid_aggregate"], name=f"Pagg RBO MPC 8"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=agg_mpc["p_grid_aggregate"], name=f"Pagg AGG MPC 8"))
        fig.add_trace(go.Scatter(x=file["parent"]["x_lims"], y=[115 for x in range(24)], name=f"Peak Threshold"))
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Power (kW)")
        fig.update_layout(title_text=f"Comparison of Aggregate Loads")
        fig.show()

def main():
    agg_params = {"alpha": [0.09, 0.1]} # set parameters from earlier runs
    mpc_params = {}
    include_runs = {"baseline", "rl_agg"}
    r = Reformat(agg_params=agg_params, include_runs=include_runs)

    r.rl2baseline()
    r.rl_thetas()
    if r.config["run_rl_agg"] or r.config["run_rbo_mpc"]: # plots the home response if the actual community response is simulated
        r.plot_single_home2("Crystal-RXXFA") # pv_battery
        # r.plot_single_home2(type="base")

        r.plot_all_homes()

if __name__ == "__main__":
    main()
