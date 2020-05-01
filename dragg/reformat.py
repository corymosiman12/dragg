import os
import sys
import json
from datetime import datetime, timedelta


import plotly.graph_objects as go
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
        self.files_to_reformat = [
            os.path.join(self.outputs_dir, x) for x in files
        ]
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
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_cool_on_opt"][0:self.hours], name="HVAC Cool Cmd"), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_heat_on_opt"][0:self.hours], name="HVAC Heat Cmd"), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_lims, y=h["wh_heat_on_opt"][0:self.hours], name="WH Heat Cmd"), row=2, col=1)
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
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_grid_opt"][0:self.hours], name="Pgrid (kW)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_load_opt"][0:self.hours], name="Pload (kW)"))
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_cool_on_opt"][0:self.hours], name="HVAC Cool Cmd"), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["hvac_heat_on_opt"][0:self.hours], name="HVAC Heat Cmd"), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=h["wh_heat_on_opt"][0:self.hours], name="WH Heat Cmd"), secondary_y=True)
            fig.add_trace(go.Scatter(x=self.x_lims, y=self.summary_data[n]["SPP"][0:self.hours], name="TOU Price ($/kWh"))

            fig.update_xaxes(title_text="Time of Day (hour)")
            fig.update_layout(title_text=f"Baseline - {name} - {type} type - horizon {horizon}")

            if 'pv' in type:
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_pv_opt"][0:self.hours], name="Ppv (kW)"))

            if 'battery' in type:
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["e_batt_opt"][0:self.hours], name="SOC (kW)"))
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_batt_ch"][0:self.hours], name="Pch (kW)"))
                fig.add_trace(go.Scatter(x=self.x_lims, y=h["p_batt_disch"][0:self.hours], name="Pdis (kW)"))

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

        fig.add_trace(go.Scatter(x=x, y=agg_cost, name="Aggregate Cost"))
        fig.add_trace(go.Scatter(x=x, y=comp_time, name="Solve Time"), secondary_y=True)

        fig.update_yaxes(title_text="Cost ($)")
        fig.update_yaxes(title_text="Solve Time (seconds)", secondary_y=True)
        fig.update_xaxes(title_text="Prediction Horizon")
        fig.update_layout(title_text=f"Baseline - {n_homes} homes")
        fig.show()

    def reward_prices_over_time(self):
        file = os.path.join(self.outputs_dir, "2015-01-01T00_2015-01-02T00-agg_mpc_all-homes_20-horizon_8-iter-results.json")
        with open(file) as f:
            data = json.load(f)

        fig = make_subplots(rows=3, cols=1)
        rp = []
        for timestep in data:
            x = [x for x in range(len(timestep['agg_cost']))]
            rp.append(timestep['reward_price'][-1])
            if len(x) > 1:
                fig.add_trace(go.Scatter(x=x, y=timestep['agg_cost'], name=f"TS: {timestep['timestep']}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=timestep['agg_load'], name=f"TS: {timestep['timestep']}"), row=2, col=1)

        x2 = [x for x in range(24)]
        fig.add_trace(go.Scatter(x=x2, y=rp, name=f"Reward Price"), row=3, col=1)

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
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_1-results.json",
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_4-results.json",
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_6-results.json",
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_8-results.json",
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_10-results.json",
        "2015-01-01T00_2015-01-02T00-baseline_all-homes_20-horizon_12-results.json",
        "2015-01-01T00_2015-01-02T00-agg_mpc_all-homes_20-horizon_8-results.json"
    ]
    r = Reformat(files)
    # r.compare_agg_between_runs(n_homes=20)
    # r.calc_agg_costs()
    # r.computation_time_vs_horizon_vs_agg_cost(n_homes=20)
    # r.plot_baseline_summary()
    # r.plot_baseline_summary2()
    # r.plot_agg_vs_homes()
    # r.plot_single_home("Jesse-PK4IH")
    # r.plot_single_home2("Jesse-PK4IH") # base
    # r.plot_single_home2("Crystal-RXXFA") # pv_battery
    # r.plot_single_home2("Dawn-L23XI") # pv_only
    # r.plot_single_home2("David-JONNO") # battery_only
    # r.reward_prices_over_time()
    # r.single_home_rbo_mpc_vs_agg_mpc("Crystal-RXXFA")
    # r.single_home_rbo_mpc_vs_agg_mpc("Jason-INS3S")
    # r.single_home_rbo_mpc_vs_agg_mpc2()
    r.baseline_vs()
