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
import dragg

class Plotter():
	def __init__(self, res_file=None, conf_file=None):
		self.res_file = res_file if res_file else self.get_res_file()
		self.conf_file = conf_file if conf_file else self.get_conf_file()
		with open(self.res_file + 'results.json') as f:
			self.data = json.load(f)

		with open(self.conf_file) as f:
			self.conf_data = json.load(f)

		self.xlims = pd.date_range(start=self.conf_data[-1]["start_dt"], end=self.conf_data[-1]["end_dt"], periods=self.conf_data[-1]["num_timesteps"]+1)

	def get_res_file(self):
		config_file = 'data/config.toml'
		with open(config_file) as f:
			config = toml.load(f)

		dt = 60 // config["agg"]["subhourly_steps"]
		sub_dt = dt // config["home"]["hems"]["sub_subhourly_steps"]
		start_dt = config["simulation"]["start_datetime"][:-3] + "T" + config["simulation"]["start_datetime"][-2:]
		end_dt = config["simulation"]["end_datetime"][:-3] + "T" + config["simulation"]["end_datetime"][-2:]
		return f'outputs/{start_dt}_{end_dt}/{config["simulation"]["check_type"]}-homes_{config["community"]["total_number_homes"]}-horizon_{config["home"]["hems"]["prediction_horizon"]}-interval_{dt}-{sub_dt}-solver_{config["home"]["hems"]["solver"]}/version-{config["simulation"]["named_version"]}/baseline/'

	def get_conf_file(self):
		config_file = 'data/config.toml'
		with open(config_file) as f:
			config = toml.load(f)

		return f'outputs/{config["simulation"]["check_type"]}_homes-{config["community"]["total_number_homes"]}-config.json'

	def plot_soc(self, name="PLAYER", debug=False):
		skip = 2 if (2 * len(self.data[name]["t_in_min"]) == len(self.data["Summary"]["OAT"])) else 1

		fig = make_subplots(rows=1, cols=3, specs=[[{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}]], subplot_titles=("Room Temp", "WH Temp", "EV SOC"))

		# plot indoor temperature
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data[name]["t_in_min"], 
					mode='lines',
    				line_color='lightskyblue',
					showlegend=False),
					row=1, col=1)
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data[name]["t_in_max"],
					fill='tonexty', 
					mode='lines',
    				line_color='lightskyblue',
					showlegend=False),
					row=1, col=1)
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data[name]["temp_in_opt"],
					mode='lines',
    				line_color='lightskyblue',
					name="Room Temp"),
					row=1, col=1)
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data["Summary"]["OAT"][::skip],
					mode='lines',
    				line_color='slateblue',
    				line_dash='dash',
					name="Outdoor Temp"),
					row=1, col=1, 
					secondary_y=False)
		fig.add_trace(go.Scatter(x=self.xlims, y=np.multiply(self.data["Summary"]["GHI"][::skip],0.001),
					mode='lines',
    				line_color='darkorange',
    				line_dash="dot",
					name="GHI"),
					row=1, col=1, 
					secondary_y=True)

		fig.update_yaxes(title_text="<b>Temperature</b> [deg C]", row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(x=self.xlims, y= np.multiply(-1,self.data[name]["hvac_cool_on_opt"]),  name="Cooling on/off", marker_color="dodgerblue"), row=1, col=1, secondary_y=True)
		fig.add_trace(go.Bar(x=self.xlims, y= self.data[name]["hvac_heat_on_opt"], name="Heat on/off", marker_color="firebrick"), row=1, col=1, secondary_y=True)
		if debug:
			fig.add_trace(go.Scatter(x=self.xlims, y=np.divide(np.cumsum(np.subtract(self.data[name]["hvac_heat_on_opt"], self.data[name]["hvac_cool_on_opt"])), 1+np.arange(self.conf_data[-1]["num_timesteps"]))), row=1, col=1, secondary_y=True)


		# plot water heater temperature
		x = False
		i = 0
		while not x:
			if self.conf_data[i]["name"] == name:
				x = True
			else:
				i += 1

		fig.add_trace(go.Scatter(x=self.xlims, y=self.conf_data[i]['wh']['temp_wh_min'] * np.ones(len(self.data[name]["temp_wh_opt"])), 
					mode='lines',
    				line_color='indianred',
					showlegend=False),
					row=1, col=2)
		fig.add_trace(go.Scatter(x=self.xlims, y=self.conf_data[i]['wh']['temp_wh_max'] * np.ones(len(self.data[name]["temp_wh_opt"])),
					fill='tonexty', 
					mode='lines',
    				line_color='indianred',
					showlegend=False),
					row=1, col=2)
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data[name]["temp_wh_opt"],
					mode='lines',
    				line_color='indianred',
					name="Hot Water Temp"),
					row=1, col=2)
		fig.update_yaxes(title_text="<b>Temperature</b> [deg C]", row=1, col=2, secondary_y=False)
		fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["wh_heat_on_opt"], name="WH on/off", marker_color="firebrick"),row=1, col=2, secondary_y=True)
		if debug:
			fig.add_trace(go.Scatter(x=self.xlims, y=np.divide(np.cumsum(self.data[name]["wh_heat_on_opt"]), 1+np.arange(self.conf_data[-1]["num_timesteps"]))), row=1, col=2, secondary_y=True)

		# plot EV state of charge
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data[name]["e_ev_opt"],
					mode='lines',
					line_color='lightseagreen', 
					name="EV SOC"),
					row=1, col=3)
		fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["p_ev_ch"], 
					name="EV chg", marker_color="seagreen"), 
					row=1, col=3, 
					secondary_y=True)
		fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["p_v2g"], 
					name="EV P2G", marker_color="seagreen"), 
					row=1, col=3, 
					secondary_y=True)
		fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["p_ev_disch"], name="EV Trip"), row=1, col=3, secondary_y=True)
		fig.update_yaxes(title_text="<b>EV SOC</b> [kWh]", row=1, col=3, secondary_y=False)

		fig.update_layout(height=600, width=1500, title_text=name)

		if debug:
			for i in range(1,4):
				fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["correct_solve"], name="correct_solve"), row=1, col=i, secondary_y=True)

		return fig 

	def plot_community_peak(self):
		total = np.zeros(self.conf_data[-1]["num_timesteps"])
		fig = go.Figure()
		for k,v in self.data.items():
			color = 'coral' if k == "PLAYER" else 'cadetblue'
			try:
				fig.add_trace(go.Scatter(x=self.xlims, y=v["p_grid_opt"], stackgroup='one', name=k, line_color=color))
				total = np.add(total, v["p_grid_opt"])
			except:
				pass

		# fig.add_trace(go.Scatter(x = self.xlims, y=np.cumsum(total)/(1+np.arange(self.conf_data[-1]["num_timesteps"]))))
		fig.update_layout(title_text="Community Demand")
		return fig 

	def main(self):
		for name in self.data.keys():
			if not name == "Summary":
				self.plot_soc(name=name).show()

		self.plot_community_peak().show()

if __name__=='__main__':
	p = Plotter()
	p.main()
