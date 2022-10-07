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
from prettytable import PrettyTable
from copy import copy, deepcopy

from dragg.logger import Logger

class Reformat:
    def __init__(self):
        self.log = Logger("reformat")

        self.data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
        self.outputs_dir = os.path.expanduser(os.environ.get('OUTPUT_DIR','outputs'))
        if not os.path.isdir(self.outputs_dir):
            self.log.logger.error("No outputs directory found.")
            quit()
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
        self.config = self._import_config()

        self.add_date_ranges()
        self.add_mpc_params()
        self.date_folders = self.set_date_folders()
        self.mpc_folders = self.set_mpc_folders()
        self.files = self.set_files()

        self.fig_list = None
        self.save_path = os.path.join('outputs', 'images', datetime.now().strftime("%m%dT%H%M%S"))

    def main(self):
        # put a list of plotting functions here
        self.sample_homes = ["Serena-98EPE","Robert-2D73X", "Crystal-RXXFA", "Myles-XQ5IA"]
        # self.plots = [self.rl2baseline] + 
        self.plots = [self.plot_single_home] #+ [self.plot_ev]

        self.images = self.plot_all()

    def plot_all(self, save_images=False):
        figs = []
        for plot in self.plots:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
#            fig.update_layout(
#                font=dict(
#                    size=65,
#                )
#            )
            fig.update_xaxes(
                title_standoff=80
            )
            fig.update_yaxes(
                title_standoff=60
            )
            fig = plot(fig)
            if isinstance(fig, list):
                for f in fig:
                    f.show()
                    figs += [f]
            else:
                fig.show()
                figs += [fig]

        return figs

    def save_images(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        for img in self.images:
            self.log.logger.info(f"Saving images of outputs to timestamped folder at {self.save_path}.")
            try:
                path = os.path.join(self.save_path, f"{img.layout.title.text}.png")
                pio.write_image(img, path, width=1024, height=768)
            except:
                self.log.logger.error("Could not save plotly image(s) to outputs directory.")

    def add_date_ranges(self):
        start_dates = set([datetime.strptime(self.config['simulation']['start_datetime'], '%Y-%m-%d %H')])
        end_dates = set([datetime.strptime(self.config['simulation']['end_datetime'], '%Y-%m-%d %H')])
        temp = {"start_datetime": start_dates, "end_datetime": end_dates}
        self.date_ranges = temp

    def add_mpc_params(self):
        n_houses = self.config['community']['total_number_homes']
        mpc_horizon = self.config['home']['hems']['prediction_horizon']
        dt = self.config['home']['hems']['sub_subhourly_steps']
        solver = self.config['home']['hems']['solver']
        check_type = self.config['simulation']['check_type']
        agg_interval = self.config['agg']['subhourly_steps']
        temp = {"n_houses": set([n_houses]), "mpc_prediction_horizons": set([mpc_horizon]), "mpc_hourly_steps": set([dt]), "check_type": set([check_type]), "agg_interval": set([agg_interval]), "solver": set([solver])}
        # for key in temp:
        #     if key in additional_params:
        #         temp[key] |= set(additional_params[key])
        self.mpc_params = temp

        self.versions = set([self.config['simulation']['named_version']])

    def set_date_folders(self):
        temp = []
        # self.date_ranges['mpc_steps'] = set([self.config['home']['hems']['sub_subhourly_steps']])
        # self.date_ranges['rl_steps'] = set([self.config['agg']['subhourly_steps']])
        keys, values = zip(*self.date_ranges.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        permutations = sorted(permutations, key=lambda i: i['end_datetime'], reverse=True)

        for i in permutations:
            date_folder = os.path.join(self.outputs_dir, f"{i['start_datetime'].strftime('%Y-%m-%dT%H')}_{i['end_datetime'].strftime('%Y-%m-%dT%H')}")
            self.log.logger.info(f"Looking for files in: {date_folder}.")
            if os.path.isdir(date_folder):
                hours = i['end_datetime'] - i['start_datetime']
                hours = int(hours.total_seconds() / 3600)

                new_folder = {"folder": date_folder, "hours": hours, "start_dt": i['start_datetime']}
                temp.append(new_folder)

        if len(temp) == 0:
            self.log.logger.error("No files found for the date ranges specified.")
            exit()

        return temp

    def set_mpc_folders(self):
        temp = []
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]
        for j in self.date_folders:
            for i in permutations:
                mpc_folder = os.path.join(j["folder"], f"{i['check_type']}-homes_{i['n_houses']}-horizon_{i['mpc_prediction_horizons']}-interval_{60 // i['agg_interval']}-{60 // i['mpc_hourly_steps'] // i['agg_interval']}-solver_{i['solver']}")
                if os.path.isdir(mpc_folder):
                    timesteps = j['hours'] * i['agg_interval']
                    minutes = 60 // i['agg_interval']
                    x_lims = [j['start_dt'] + timedelta(minutes=minutes*x) for x in range(timesteps)]

                    set = {'path': mpc_folder, 'agg_dt': i['agg_interval'], 'ts': timesteps, 'x_lims': x_lims,}
                    if not mpc_folder in temp:
                        temp.append(set)
        for x in temp:
            print(x['path'])
        return temp

    def set_files(self):
        temp = []
        keys, values = zip(*self.mpc_params.items())
        permutations = [dict(zip(keys, v)) for v in it.product(*values)]

        color_families = [['rgb(204,236,230)','rgb(153,216,201)','rgb(102,194,164)','rgb(65,174,118)','rgb(35,139,69)','rgb(0,88,36)'],
                        ['rgb(191,211,230)','rgb(158,188,218)','rgb(140,150,198)','rgb(140,107,177)','rgb(136,65,157)','rgb(110,1,107)'],
                        ['rgb(217,217,217)','rgb(189,189,189)','rgb(150,150,150)','rgb(115,115,115)','rgb(82,82,82)','rgb(37,37,37)'],
                        ['rgb(253,208,162)','rgb(253,174,107)','rgb(253,141,60)','rgb(241,105,19)','rgb(217,72,1)','rgb(140,45,4)'],]
        c = 0
        d = 0
        dash = ["solid", "dash", "dot", "dashdot"]
        for j in self.mpc_folders:
            path = j['path']
            for i in permutations:
                for k in self.versions:
                    dir = os.path.join(path, f"version-{k}")
                    for case_dir in os.listdir(dir):
                        file = os.path.join(dir, case_dir, "results.json")
                        if os.path.isfile(file):
                            name = f"{case_dir}, v = {k}"
                            set = {"results": file, "name": name, "parent": j, "color": color_families[c][d], "dash":dash[c]}
                            temp.append(set)
                            self.log.logger.info(f"Adding baseline file at {file}")
                            d = (d + 1) % len(color_families[c])
                        c = (c + 1) % len(color_families)

        return temp

    def get_type_list(self, type):
        type_list = set([])
        i = 0
        for file in self.files:
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

        self.log.logger.info(f"{len(type_list)} homes found of type {type}: {type_list}")
        return type_list

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)

        with open(self.config_file, 'r') as f:
            data = toml.load(f)

        return data

    def plot_environmental_values(self, name, fig, summary, file, fname):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["OAT"][0:file["parent"]["ts"]], name=f"OAT (C)", visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["GHI"][0:file["parent"]["ts"]], name=f"GHI", line={'color':'goldenrod', 'width':8}, visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=summary["TOU"][0:file["parent"]["ts"]], name=f"TOU Price ($/kWh)", line_shape='hv', visible='legendonly'), secondary_y=True)
        fig = self.plot_thermal_bounds(fig, file['parent']['x_lims'], name, fname)
        return fig

    def plot_thermal_bounds(self, fig, x_lims, name, fname):
        ah_file = os.path.join(self.outputs_dir, f"all_homes-{self.config['community']['total_number_homes']}-config.json")
        with open(ah_file) as f:
            data = json.load(f)

        for dict in data:
            if dict['name'] == name:
                data = dict

#        fig.add_trace(go.Scatter(x=x_lims, y=data['hvac']['temp_in_min'] * np.ones(len(x_lims)), name=f"Tin_min", fill=None, showlegend=False, mode='lines', line_color='lightsteelblue'))
#        fig.add_trace(go.Scatter(x=x_lims, y=data['hvac']['temp_in_max'] * np.ones(len(x_lims)), name=f"Tin_bounds", fill='tonexty' , mode='lines', line_color='lightsteelblue'))

        fig.add_trace(go.Scatter(x=x_lims, y=data['wh']['temp_wh_min'] * np.ones(len(x_lims)), name=f"Twh_min", fill=None, showlegend=False, mode='lines', line_color='pink'))
        fig.add_trace(go.Scatter(x=x_lims, y=data['wh']['temp_wh_max'] * np.ones(len(x_lims)), name=f"Twh_bounds", fill='tonexty' , mode='lines', line_color='pink'))
        return fig

    def plot_base_home(self, name, fig, data, summary, fname, file, plot_price=True):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["temp_in_opt"], name=f"Tin - {fname}", legendgroup='tin', line={'color':'blue', 'width':8, 'dash':file['dash']}))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["temp_wh_opt"], showlegend=True, legendgroup='twh', name=f"Twh - {fname}", line={'color':'firebrick', 'width':8, 'dash':file['dash']}))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data['t_in_min'], name=f"Tin_min", fill=None, showlegend=False, mode='lines', line_color='lightsteelblue',line_shape='hv'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data['t_in_max'], name=f"Tin_bounds", fill='tonexty' , mode='lines', line_color='lightsteelblue',line_shape='hv'))
#        fig.update_layout(legend=dict(
#            yanchor="top",
#            y=0.99,
#            xanchor="left",
#            x=0.03,
##            font=dict(
##                size=65),
##            ),
#            yaxis_title="Temperature (deg C)"
#        )

        return fig

    def plot_pv(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_pv_opt"], name=f"Ppv (kW)", line_color='orange', line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["u_pv_curt_opt"], name=f"U_pv_curt (kW) - {fname}", line_shape='hv', visible='legendonly'))
        return fig

    def plot_battery(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["e_batt_opt"], name=f"SOC (kWh) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_batt_ch"], name=f"Pch (kW) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_batt_disch"], name=f"Pdis (kW) - {fname}", line_shape='hv', visible='legendonly'))
        return fig

    def plot_ev(self, name, fig, data, fname, file):
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["e_ev_opt"], name=f"SOC (kWh) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_ev_ch"], name=f"Pch (kW) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_ev_disch"], name=f"Pdis (kW) - {fname}", line_shape='hv', visible='legendonly'))
        fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["p_v2g"], name=f"Pv2g (kW) - {fname}", line_shape='hv', visible='legendonly'))
        return fig

    def plot_single_home(self, fig):
        figs = [copy(fig) for _ in self.sample_homes]
        #for self.sample_home in self.sample_homes:
        for i in range(len(self.sample_homes)):
            self.sample_home = self.sample_homes[i]
            fig = figs[i]
            if self.sample_home is None:
                if type is None:
                    type = "base"
                    self.log.logger.warning("Specify a home type or name. Proceeding with home of type: \"base\".")

                type_list = self._type_list(type)
                self.sample_home = random.sample(type_list,1)[0]
                self.log.logger.info(f"Proceeding with home: {name}")

            flag = False
            for file in self.files:
                with open(file["results"]) as f:
                    comm_data = json.load(f)

                try:
                    data = comm_data[self.sample_home]
                except:
                    self.log.logger.error(f"No home with name: {self.sample_home}")
                    return

                type = data["type"]
                summary = comm_data["Summary"]

                if not flag:
                    fig = self.plot_environmental_values(self.sample_home, fig, summary, file, file["name"])
                    flag = True

                fig.update_xaxes(title_text="Time of Day (hour)")
                fig.update_layout(title_text=f"{self.sample_home} - {type} type")

                fig = self.plot_base_home(self.sample_home, fig, data, summary, file["name"], file)

                if 'pv' in type:
                    fig = self.plot_pv(self.sample_home, fig, data, file["name"], file)

                if 'batt' in type:
                    fig = self.plot_battery(self.sample_home, fig, data, file["name"], file)

                fig = self.plot_ev(self.sample_home, fig, data, file['name'], file)
                #figs += [fig]
        return figs

    def plot_all_homes(self, fig=None):
        homes = ["Crystal-RXXFA","Myles-XQ5IA","Lillie-NMHUH","Robert-2D73X","Serena-98EPE","Gary-U95TS","Bruno-PVRNB","Dorothy-9XMNY","Jason-INS3S","Alvin-4BAYB",]
        for self.sample_home in homes:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
#            fig.update_layout(
#                font=dict(
#                    size = 12
#                )
#            )
            fig = self.plot_single_home(fig)

        return

    def plot_baseline(self, fig):
        for file in self.files:
            with open(file["results"]) as f:
                data = json.load(f)

            ts = len(data['Summary']['p_grid_aggregate'])-1
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_aggregate"], name=f"Agg Load - {file['name']}", line_shape='hv', line={'color':file['color'], 'width':4, 'dash':'solid'}))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(np.divide(data["Summary"]["p_grid_aggregate"], file['parent']['agg_dt'])), name=f"Cumulative Agg Load - {file['name']}", line_shape='hv', visible='legendonly', line={'color':file['color'], 'width':4, 'dash':'dash'}))
            fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.divide(np.cumsum(data["Summary"]["p_grid_aggregate"]), np.arange(ts + 1) + 1), name=f"Avg Cumulative Agg Load - {file['name']}", line_shape='hv', visible='legendonly', line={'color':file['color'], 'width':4, 'dash':'dashdot'}))
        return fig

    def plot_typ_day(self, fig):
        rl_counter = 0
        tou_counter = 0
        dn_counter = 0
        for file in self.files:
            flag = True

            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]

            ts = len(data['Summary']['p_grid_aggregate'])-1
            rl_setpoint = data['Summary']['p_grid_setpoint']
            if 'clipped' in file['name']:
                rl_setpoint = np.clip(rl_setpoint, 45, 60)
            loads = np.array(data["Summary"]["p_grid_aggregate"])
            loads = loads[:len(loads) // (24*file['parent']['agg_dt']) * 24 * file['parent']['agg_dt']]
            if len(loads) > 24:
                daily_max_loads = np.repeat(np.amax(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_min_loads = np.repeat(np.amin(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_range_loads = np.subtract(daily_max_loads, daily_min_loads)
                daily_range_loads = [abs(loads[max(i-6, 0)] - loads[min(i+6, len(loads)-1)]) for i in range(len(loads))]
                daily_avg_loads = np.repeat(np.mean(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_std_loads = np.repeat(np.std(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_std_loads = [np.std(loads[max(i-6, 0):i+6]) for i in range(len(loads))]

                composite_day = np.average(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=0)
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=composite_day, name=f"{name}", opacity=0.5, showlegend=flag, line={'color':clr, 'width':8, 'dash':dash}))

                fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.45,
                    xanchor="left",
                    x=0.7
                ))

            fig.update_layout(
#                font=dict(
#                    # family="Courier New, monospace",
#                    size=65,
#                ),
                title="Avg Daily Load Profile",
                xaxis_title="Time of Day",
                yaxis_title="Agg. Demand (kW)"
            )

            fig.update_xaxes(
                title_standoff=80
            )
            fig.update_yaxes(
                title_standoff=60
            )

        return fig

    def plot_max_and_12hravg(self, fig):
        for file in self.files:
            # all_avgs.add_column()
            clr = file['color']

            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]
            ts = len(data['Summary']['p_grid_aggregate'])-1
            rl_setpoint = data['Summary']['p_grid_setpoint']
            if 'clipped' in file['name']:
                rl_setpoint = np.clip(rl_setpoint, 45, 60)
            loads = np.array(data["Summary"]["p_grid_aggregate"])
            loads = loads[:len(loads) // (24*file['parent']['agg_dt']) * 24 * file['parent']['agg_dt']]
            if len(loads) > 24:
                daily_max_loads = np.repeat(np.amax(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_min_loads = np.repeat(np.amin(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_range_loads = np.subtract(daily_max_loads, daily_min_loads)
                daily_range_loads = [abs(loads[max(i-6, 0)] - loads[min(i+6, len(loads)-1)]) for i in range(len(loads))]
                daily_avg_loads = np.repeat(np.mean(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_std_loads = np.repeat(np.std(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])

                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_max_loads, name=f"{name} - Daily Max", line_shape='hv', opacity=1, legendgroup="first", line={'color':'firebrick', 'dash':dash, 'width':8}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=rl_setpoint, name=f"{name} - 12 Hr Avg", opacity=0.5, legendgroup="second", line={'color':'blue', 'dash':dash, 'width':8}))

                fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.8,
                    xanchor="left",
                    x=0.7
                ))

            fig.update_layout(
#                font=dict(
#                    size=65,
#                ),
                title="12 Hour Avg and Daily Max",
                yaxis_title="Agg. Demand (kW)"
            )

            fig.update_xaxes(
                title_standoff=80
            )
            fig.update_yaxes(
                title_standoff=60
            )

        return fig


    def plot_parametric(self, fig):
        all_daily_stats = PrettyTable(['run name', 'avg daily max', 'std daily max','overall max', 'avg daily range'])
        for file in self.files:
            clr = file['color']

            with open(file["results"]) as f:
                data = json.load(f)

            name = file["name"]
            ts = len(data['Summary']['p_grid_aggregate'])-1
            rl_setpoint = data['Summary']['p_grid_setpoint']
            if 'clipped' in file['name']:
                rl_setpoint = np.clip(rl_setpoint, 45, 60)
            loads = np.array(data["Summary"]["p_grid_aggregate"])
            loads = loads[:len(loads) // (24*file['parent']['agg_dt']) * 24 * file['parent']['agg_dt']]
            if len(loads) >= 24:
                daily_max_loads = np.repeat(np.amax(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_min_loads = np.repeat(np.amin(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_range_loads = np.subtract(daily_max_loads, daily_min_loads)
                daily_range_loads = [abs(loads[max(i-6, 0)] - loads[min(i+6, len(loads)-1)]) for i in range(len(loads))]
                daily_avg_loads = np.repeat(np.mean(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=1), 24*file['parent']['agg_dt'])
                daily_std_loads = [np.std(loads[max(i-6, 0):i+6]) for i in range(len(loads))]

                composite_day = np.average(loads.reshape(-1, 24*file['parent']['agg_dt']), axis=0)
                fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.45,
                    xanchor="left",
                    x=0.5
                ))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=rl_setpoint, name=f"{name} - 12 Hr Avg", opacity=0.5, line={'color':clr, 'width':8}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=data["Summary"]["p_grid_aggregate"], name=f"Agg Load - RL - {name}", line_shape='hv', line={'color':clr}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_max_loads, name=f"{name} - Daily Max", line_shape='hv', opacity=0.5, line={'color':clr, 'dash':'dot'}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_min_loads, name=f"Daily Min Agg Load - RL - {name}", line_shape='hv', opacity=0.5, line={'color':clr, 'dash':'dash'}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_range_loads, name=f"Daily Agg Load Range - RL - {name}", line_shape='hv', opacity=0.5, line={'color':clr}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.average(daily_range_loads) * np.ones(len(loads)), name=f"Avg Daily Agg Load Range - RL - {name}", line_shape='hv', opacity=0.5, line={'color':clr}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_avg_loads, name=f"Daily Avg Agg Load - RL - {name}", line_shape='hv', opacity=0.5, line={'color':clr, 'dash':'dash'}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=daily_std_loads, name=f"Daily Std Agg Load - RL - {name}", line_shape='hv',  opacity=0.5, line={'color':clr, 'dash':'dashdot'}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.average(daily_std_loads) * np.ones(len(loads)), name=f"Avg Daily Std Agg Load - RL - {name}", line_shape='hv',  opacity=0.5, line={'color':clr, 'dash':'dashdot'}))
                fig.add_trace(go.Scatter(x=file['parent']['x_lims'], y=np.cumsum(np.divide(data["Summary"]["p_grid_aggregate"],file['parent']['agg_dt'])), name=f"{name}", line_shape='hv', visible='legendonly', line={'color':clr, }))
                all_daily_stats.add_row([file['name'], np.average(daily_max_loads), np.std(daily_max_loads), max(daily_max_loads), np.average(daily_range_loads)])
        else:
            self.log.logger.warning("Not enough data collected to have daily stats, try running the aggregator for longer.")
        print(all_daily_stats)
        return fig

    def rl2baseline(self, fig):
        if len(self.files) == 0:
            self.log.logger.warning("No aggregator runs found for analysis.")
            return fig
        fig = self.plot_baseline(fig)
        fig = self.plot_parametric(fig)
        fig.update_layout(title_text="RL Baseline Comparison")
        fig.update_layout(
            title="Avg Daily Load Profile",
            xaxis_title="Time of Day",
            yaxis_title="Agg. Demand (kWh)",)
        return fig

    def all_rps(self, fig):
        for file in self.files:
            with open(file['results']) as f:
                data = json.load(f)

            rps = data['Summary']['RP']
            fig.add_trace(go.Histogram(x=rps, name=f"{file['name']}"), row=1, col=1)

            with open(file['q_results']) as f:
                data = json.load(f)
            data = data["horizon"]
            mu = np.array(data["mu"])
            std = self.config['agg']['parameters']['exploration_rate'][0]
            delta = np.subtract(mu, rps)

            fig.add_trace(go.Histogram(x=delta, name=f"{file['name']}"), row=2, col=1)
            fig.add_trace(go.Scatter(x=[-std, -std, std, std], y=[0, 0.3*len(rps), 0.3*len(rps), 0], fill="toself"), row=2, col=1)
        return fig

if __name__ == "__main__":
    r = Reformat()
    r.main()
