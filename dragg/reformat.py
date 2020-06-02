import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dragg.reformat_logger import ReformatLogger

class Reformat:
    def __init__(self, files):
        self.ref_log = ReformatLogger()
        self.data_dir = 'data'
        self.outputs_dir = 'outputs'
        if not os.path.isdir(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        self.config_file = os.path.join(self.data_dir, os.environ.get('CONFIG_FILE', 'config.json'))
        self.config = self._import_config()
        self.num_homes = self.config["total_number_homes"]
        self.check_type = self.config["check_type"]
        self.pred_horizons = self.config["prediction_horizons"]
        self.start_dt = datetime.strptime(self.config["start_datetime"], '%Y-%m-%d %H')
        self.end_dt = datetime.strptime(self.config["end_datetime"], '%Y-%m-%d %H')
        self.hours = self.end_dt - self.start_dt
        self.hours = int(self.hours.total_seconds() / 3600)
        self.files_to_reformat = files
        self.data = self._import_data()
        self.summary_data = self._config_summary()
        self.x_lims = [self.start_dt + timedelta(hours=x) for x in range(self.hours)]

    def _import_config(self):
        if not os.path.exists(self.config_file):
            self.ref_log.logger.error(f"Configuration file does not exist: {self.config_file}")
            sys.exit(1)
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        return data

    def _import_data(self):
        data = []
        for f in self.files_to_reformat:
            with open(f, 'r') as fh:
                d = json.load(fh)
                data.append(d)

        return data

    def _config_summary(self):
        s = []
        for d in self.data:
            s.append(d["Summary"])
        return s

    def plot_baseline_summary(self):
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["p_grid_aggregate"], name="Pagg (kW)"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["SPP"], name="Energy Cost ($/kWh)"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["OAT"], name="OAT (C)"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["GHI"], name="GHI (W/m2)"),
                      row=1, col=2)
        #
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_baseline_summary2(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["p_grid_aggregate"], name="Pagg (kW)"))
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["OAT"], name="OAT (C)"), secondary_y=True)
        # fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["SPP"], name="Energy Cost ($/kWh)"))
        # fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["GHI"], name="GHI (W/m2"))

        fig.update_yaxes(title_text='Power (kW)', secondary_y=False)
        fig.update_yaxes(title_text='Temperature (C)', secondary_y=True)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_agg_vs_homes(self, names):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["p_grid_aggregate"], name="Pagg (kW)"))
        for home, data in self.baseline_data.items():
            # if home in names:
            fig.add_trace(go.Scatter(x=self.x_lims, y=data["p_grid_opt"], opacity=0.25, name=f"{home}-{data['type']}"), secondary_y=True)

        fig.update_yaxes(title_text='Pagg (kW)', secondary_y=False)
        fig.update_yaxes(title_text='Pload (kW)', secondary_y=True)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {self.num_homes} Homes")

        fig.show()

    def plot_single_home(self, name):
        h = self.baseline_data.get(name, None)
        if h is None:
            self.ref_log.logger.error(f"No home with name: {name}")
            return
        type = h["type"]
        fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True}, {}],
                                                   [{}, {}]])
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["temp_in_opt"][0:self.hours], name="Tin (C)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["temp_wh_opt"][0:self.hours], name="Twh (C)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["OAT"][0:self.hours], name="OAT (C)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["GHI"][0:self.hours], name="GHI (W/m2)"), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_grid_opt"][0:self.hours], name="Pgrid (kW)"), row=1, col=2)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_load_opt"][0:self.hours], name="Pload (kW)"), row=1, col=2)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_cool_on_opt"][0:self.hours], name="HVAC Cool Cmd", line_shape='hv'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_heat_on_opt"][0:self.hours], name="HVAC Heat Cmd", line_shape='hv'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["wh_heat_on_opt"][0:self.hours], name="WH Heat Cmd", line_shape='hv'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary["SPP"][0:self.hours], name="TOU Price ($/kWh"), row=2, col=2)

        fig.update_yaxes(title_text="Temperature (C)", row=1, col=1)
        fig.update_yaxes(title_text="GHI (W/m2)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
        fig.update_yaxes(title_text="CMD Signals", row=2, col=1)
        fig.update_yaxes(title_text="Cost ($/kWh)", row=2, col=2)
        fig.update_xaxes(title_text="Time of Day (hour)")
        fig.update_layout(title_text=f"Baseline - {name} - {type} type")
        fig.show()

        if type == "pv_only":
            self.plot_single_home_pv(h, name)
        elif type == "battery_only":
            self.plot_single_home_battery(h, name)
        elif type == "pv_battery":
            self.plot_single_home_pv_battery(h, name)

    def plot_single_home2(self, name):
        for n, d in enumerate(self.data):
            h = d.get(name, None)
            if h is None:
                self.ref_log.logger.error(f"No home with name: {name}")
                return
            type = h["type"]
            horizon = d["Summary"]["horizon"]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["temp_in_opt"][0:self.hours], name="Tin (C)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["temp_wh_opt"][0:self.hours], name="Twh (C)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary_data[n]["OAT"][0:self.hours], name="OAT (C)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary_data[n]["GHI"][0:self.hours], name="GHI (W/m2)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_grid_opt"][0:self.hours], name="Pgrid (kW)", line_shape='hv'))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_load_opt"][0:self.hours], name="Pload (kW)", line_shape='hv'))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_cool_on_opt"][0:self.hours], name="HVAC Cool Cmd", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_heat_on_opt"][0:self.hours], name="HVAC Heat Cmd", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["wh_heat_on_opt"][0:self.hours], name="WH Heat Cmd", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary_data[n]["TOU"][0:self.hours], name="TOU Price ($/kWh)", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=np.add(self.summary_data[n]["TOU"][0:self.hours], self.summary_data[n]["RP"][0:self.hours]), name="Actual Price ($/kWh)", line_shape='hv'), secondary_y=True)

            case = self.summary_data[n]["case"]
            fig.update_xaxes(title_text="Time of Day (hour)")
            fig.update_layout(title_text=f"{case} - {name} - {type} type - horizon {horizon}")

            if 'pv' in type:
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_pv_opt"][0:self.hours], name="Ppv (kW)", line_shape='hv'))

            if 'battery' in type:
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["e_batt_opt"][0:self.hours], name="SOC (kW)", line_shape='hv'))
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_batt_ch"][0:self.hours], name="Pch (kW)", line_shape='hv'))
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_batt_disch"][0:self.hours], name="Pdis (kW)", line_shape='hv'))

            fig.show()

    def compare_agg_between_runs(self, n_homes):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for n, s in enumerate(self.summary_data):
            if n == 0:
                fig.add_trace(go.Scatter(x=self.x_lims, y=s["SPP"], name=f"TOU Price"), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=s["p_grid_aggregate"], name=f"Pagg H: {s['horizon']}"))

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

        fig.add_trace(go.Scatter(x=self.x_lims, y=agg_cost, name="Aggregate Cost"))
        fig.add_trace(go.Scatter(x=self.x_lims, y=comp_time, name="Solve Time"), secondary_y=True)

        fig.update_yaxes(title_text="Cost ($)")
        fig.update_yaxes(title_text="Solve Time (seconds)", secondary_y=True)
        fig.update_xaxes(title_text="Prediction Horizon")
        fig.update_layout(title_text=f"Baseline - {n_homes} homes")
        fig.show()

    def rl_reward_prices(self):
        file = os.path.join(self.outputs_dir, "rl_agg", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-iter-results.json")
        rlagg = os.path.join(self.outputs_dir, "rl_agg", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")
        baseline = os.path.join(self.outputs_dir, "baseline", "2015-01-01T00_2015-01-10T00-baseline_all-homes_20-horizon_8-results.json")

        with open(file) as f:
            data = json.load(f)

        with open(baseline) as f:
            baselinedata = json.load(f)

        with open(rlagg) as f:
            rlaggdata = json.load(f)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # fig = make_subplots()
        rp = []
        shapes = []
        al = []
        sp = []
        prev = False
        for timestep in data:
            rp.append(timestep['reward_price'])
            if timestep['is_greedy'] and not prev:
                temp = {'type':'rect',
                        'xref':"x",
                        'yref':"paper",
                        'fillcolor':"LightSalmon",
                        'opacity':0.2,
                        'layer':"below",
                        'line_width':0,
                        'x0':self.start_dt + timedelta(hours=timestep["timestep"]),
                        'y0':0}
            elif not timestep['is_greedy'] and prev:
                temp["x1"] = self.start_dt + timedelta(hours=timestep["timestep"])
                temp["y1"] = 1
                shapes.append(temp)
            elif timestep["timestep"] == self.hours - 1 and prev:
                temp["x1"] = self.start_dt + timedelta(hours=timestep["timestep"])
                temp["y1"] = 1
                shapes.append(temp)
            prev = timestep["is_greedy"]
            al.append(timestep['agg_load'])

        fig.update_layout(shapes=shapes)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rp, name="Reward Price", line_shape='hv'))
        fig.add_trace(go.Scatter(x=self.x_lims, y=al, name="Aggregate Load"), secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rlaggdata["Summary"]["p_grid_setpoint"], name="Setpoint Load"), secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=baselinedata["Summary"]["p_grid_aggregate"], name="Baseline Load"), secondary_y=True)

        fig.show()

    def _show_greedy(self, fig, data):
        shapes = []

        prev = False
        for timestep in data:
            if timestep['is_greedy'] and not prev:
                temp = {'type':'rect',
                        'xref':"x",
                        'yref':"paper",
                        'fillcolor':"LightSalmon",
                        'opacity':0.2,
                        'layer':"below",
                        'line_width':0,
                        'x0':self.start_dt + timedelta(hours=timestep["timestep"]),
                        'y0':0}
            elif not timestep['is_greedy'] and prev:
                temp["x1"] = self.start_dt + timedelta(hours=timestep["timestep"])
                temp["y1"] = 1
                shapes.append(temp)
            elif timestep["timestep"] == self.hours - 1 and prev:
                temp["x1"] = self.start_dt + timedelta(hours=timestep["timestep"])
                temp["y1"] = 1
                shapes.append(temp)
            prev = timestep["is_greedy"]

        fig.update_layout(shapes = shapes)
        return fig

    def add_baseline(self, fig, baselineMPC, baselineNoMPC):
        fig.add_trace(go.Scatter(x=self.x_lims, y=baselineMPC["Summary"]["p_grid_aggregate"], name="MPC No agg."))
        fig.add_trace(go.Scatter(x=self.x_lims, y=baselineNoMPC["Summary"]["p_grid_aggregate"], name="No MPC No agg."))
        return fig

    def rl2baseline(self, rl_file, rl_qfile, baselineMPC_file, baselineNoMPC_file):

        with open(rl_file) as f:
            rldata = json.load(f)

        with open(rl_qfile) as f:
            rl_qdata = json.load(f)
        try:
            with open(baselineMPC_file) as f:
                baselineMPC = json.load(f)
        except:
            baselineMPC = None
        try:
            with open(baselineNoMPC_file) as f:
                baselineNoMPC = json.load(f)
        except:
            baselineMPC = None

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        self._show_greedy(fig, rl_qdata)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rldata["Summary"]["p_grid_aggregate"][1:], name="RL"))
        if baselineMPC and baselineNoMPC:
            self.add_baseline(fig, baselineMPC, baselineNoMPC)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rldata["Summary"]["TOU"], name="TOU Price ($/kWh)", line_shape='hv'), secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rldata["Summary"]["RP"], name="Reward Price ($/kWh)", line_shape='hv'), secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=np.add(rldata["Summary"]["TOU"], rldata["Summary"]["RP"]), name="Actual Price ($/kWh)", line_shape='hv'), secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rldata["Summary"]["p_grid_setpoint"], name="RL Setpoint Load"))
        fig.show()

    def compare_methods(self):
        base = os.path.join(self.outputs_dir, "baseline", "2015-01-01T00_2015-01-10T00-baseline_all-homes_20-horizon_8-results.json")
        basetou = os.path.join(self.outputs_dir, "baseline", "2015-01-01T00_2015-01-10T00-baseline_all-homes_20-horizon_8-results-tou.json")
        rr_rl =  os.path.join(self.outputs_dir, "rl_agg","ridge", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")
        en_rl =  os.path.join(self.outputs_dir, "rl_agg","elasticNet", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")
        la_rl =  os.path.join(self.outputs_dir, "rl_agg","lasso", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")

        files = [base, basetou, rr_rl, en_rl, la_rl]
        for i in files:
            with open(i):
                data = json.load(f)

    def compare2(self):
        opt1 = os.path.join(self.outputs_dir, "rl_agg","elasticNet", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")
        opt2 = os.path.join(self.outputs_dir, "rl_agg", "2015-01-01T00_2015-01-10T00-rl_agg_all-homes_20-horizon_8-results.json")
        baselinefile = os.path.join(self.outputs_dir, "baseline", "2015-01-01T00_2015-01-10T00-baseline_all-homes_20-horizon_1-results.json")
        with open(baselinefile) as f:
            baselinedata = json.load(f)


        opts = [opt1, opt2]
        names = ["L2 Norm", "Gradient Penalty"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=self.x_lims, y=baselinedata["Summary"]["p_grid_aggregate"], name="MPC No agg."))
        for i in range(2):
            with open(opts[i]) as f:
                data = json.load(f)

            fig.update_layout(shapes = self._show_greedy(os.path.join(self.outputs_dir, "rl_agg", "2015-01-01T00_2015-01-02T00-rl_agg_all-homes_20-horizon_8-iter-results.json")))
            fig.add_trace(go.Scatter(x=self.x_lims, y=data["Summary"]["p_grid_aggregate"], name=f"RL - {names[i]}"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=data["Summary"]["TOU"], name=f"TOU Price ($/kWh) - {names[i]}", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=data["Summary"]["RP"], name=f"Reward Price ($/kWh) - {names[i]}", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=np.add(data["Summary"]["TOU"], data["Summary"]["RP"]), name=f"Actual Price ($/kWh) - {names[i]}", line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=data["Summary"]["p_grid_setpoint"], name=f"RL Setpoint Load - {names[i]}"))

        fig.show()

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
                fig.add_trace(go.Scatter(x=self.x_lims, y=timestep['agg_cost'], name=f"TS: {timestep['timestep']}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=self.x_lims, y=timestep['agg_load'], name=f"TS: {timestep['timestep']}"), row=2, col=1)

        x2 = [x for x in range(24)]
        fig.add_trace(go.Scatter(x=self.x_lims2, y=rp, name=f"Reward Price"), row=3, col=1)

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
        fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["temp_in_opt"][0:self.hours], name="RBO MPC Tin"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=agg["temp_in_opt"][0:self.hours], name="AGG MPC Tin"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["hvac_heat_on_opt"][0:self.hours], name="RBO MPC HVAC Heat CMD"), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=agg["hvac_heat_on_opt"][0:self.hours], name="AGG MPC HVAC Heat CMD"), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["temp_wh_opt"][0:self.hours], name="RBO MPC Twh"), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=agg["temp_wh_opt"][0:self.hours], name="AGG MPC Twh"), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["wh_heat_on_opt"][0:self.hours], name="RBO MPC WH Heat CMD"), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=self.x_lims, y=agg["wh_heat_on_opt"][0:self.hours], name="AGG MPC WH Heat CMD"), row=2, col=1, secondary_y=True)
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
                fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["temp_in_opt"][0:self.hours], name="RBO MPC Tin"), row=1,
                              col=1)
                fig.add_trace(go.Scatter(x=self.x_lims, y=agg["temp_in_opt"][0:self.hours], name="AGG MPC Tin"), row=1,
                              col=1)
                fig.add_trace(
                    go.Scatter(x=self.x_lims, y=rbo["hvac_heat_on_opt"][0:self.hours], name="RBO MPC HVAC Heat CMD"), row=1,
                    col=1, secondary_y=True)
                fig.add_trace(
                    go.Scatter(x=self.x_lims, y=agg["hvac_heat_on_opt"][0:self.hours], name="AGG MPC HVAC Heat CMD"), row=1,
                    col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["temp_wh_opt"][0:self.hours], name="RBO MPC Twh"), row=2,
                              col=1)
                fig.add_trace(go.Scatter(x=self.x_lims, y=agg["temp_wh_opt"][0:self.hours], name="AGG MPC Twh"), row=2,
                              col=1)
                fig.add_trace(go.Scatter(x=self.x_lims, y=rbo["wh_heat_on_opt"][0:self.hours], name="RBO MPC WH Heat CMD"),
                              row=2, col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=self.x_lims, y=agg["wh_heat_on_opt"][0:self.hours], name="AGG MPC WH Heat CMD"),
                              row=2, col=1, secondary_y=True)
                fig.add_trace(
                    go.Scatter(x=self.x_lims, y=rbo["p_grid_opt"][0:self.hours], name="RBO MPC Pgrid"),
                    row=3, col=1)
                fig.add_trace(
                    go.Scatter(x=self.x_lims, y=agg["p_grid_opt"][0:self.hours], name="AGG MPC Pgrid"),
                    row=3, col=1)
                if 'battery' in rbo['type']:
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=rbo["p_batt_disch"][0:self.hours], name="RBO MPC Pdis"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=agg["p_batt_disch"][0:self.hours], name="AGG MPC Pdis"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=rbo["p_batt_ch"][0:self.hours], name="RBO MPC Pch"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=agg["p_batt_ch"][0:self.hours], name="AGG MPC Pch"),
                        row=3, col=1)
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=rbo["e_batt_opt"][0:self.hours], name="RBO MPC Ebatt"),
                        row=4, col=1)
                    fig.add_trace(
                        go.Scatter(x=self.x_lims, y=agg["e_batt_opt"][0:self.hours], name="AGG MPC Ebatt"),
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
        fig.add_trace(go.Scatter(x=self.x_lims, y=baseline["p_grid_aggregate"], name=f"Pagg Baseline"))
        fig.add_trace(go.Scatter(x=self.x_lims, y=rbo_mpc["p_grid_aggregate"], name=f"Pagg RBO MPC 8"))
        fig.add_trace(go.Scatter(x=self.x_lims, y=agg_mpc["p_grid_aggregate"], name=f"Pagg AGG MPC 8"))
        fig.add_trace(go.Scatter(x=self.x_lims, y=[115 for x in range(24)], name=f"Peak Threshold"))
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Power (kW)")
        fig.update_layout(title_text=f"Comparison of Aggregate Loads")
        fig.show()


if __name__ == "__main__":
    # names = ["Jesse-PK4IH", "Crystal-RXXFA", "Dawn-L23XI", "David-JONNO"]
    files = [
        os.path.join("outputs", "rl_agg", "2015-01-01T00_2015-01-08T00-rl_agg_all-homes_20-horizon_8-results.json"),
    ]
    r = Reformat(files)
    # r.compare_agg_between_runs(n_homes=20)
    # r.calc_agg_costs()
    # r.computation_time_vs_horizon_vs_agg_cost(n_homes=20)
    # r.plot_baseline_summary()
    # r.plot_baseline_summary2()
    # r.plot_agg_vs_homes()
    # r.plot_single_home("Jesse-PK4IH")
    # r.rltheta()
    r.plot_single_home2("Ruth-1HV86") # base
    r.plot_single_home2("Crystal-RXXFA") # pv_battery
    r.plot_single_home2("Dawn-L23XI") # pv_only
    r.plot_single_home2("Jason-INS3S") # battery_only
    # r.reward_prices_over_time()
    # r.rl_reward_prices()

    rlfile = os.path.join(r.outputs_dir, "rl_agg", "2015-01-01T00_2015-01-30T00-rl_agg_all-homes_20-horizon_8-results.json")
    mpc_noaggfile = os.path.join(r.outputs_dir, "baseline", "2015-01-01T00_2015-01-12T00-baseline_all-homes_20-horizon_8-results.json")
    mpc_noaggfile = None
    baselinefile = os.path.join(r.outputs_dir, "baseline", "2015-01-01T00_2015-01-12T00-baseline_all-homes_20-horizon_1-results.json")

    r.rl2baseline(rlfile, mpc_noaggfile, baselinefile)
    # r.rl2baseline_shifted()
    # r.rl_reward_prices()
    # r.single_home_rbo_mpc_vs_agg_mpc("Crystal-RXXFA")
    # r.single_home_rbo_mpc_vs_agg_mpc("Jason-INS3S")
    # r.single_home_rbo_mpc_vs_agg_mpc2()
    # r.baseline_vs()
