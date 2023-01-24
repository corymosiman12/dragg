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
	def __init__(self):
		self.res_file = 'outputs/2003-06-01T00_2003-07-01T00/all-homes_5-horizon_4-interval_60-10-solver_GLPK_MI/version-final/baseline/results.json'
		self.conf_file = 'outputs/all_homes-5-config.json'
		with open(self.res_file) as f:
			self.data = json.load(f)

		with open(self.conf_file) as f:
			self.conf_data = json.load(f)

		self.xlims = pd.date_range(start=self.conf_data[-1]["start_dt"], end=self.conf_data[-1]["end_dt"], periods=self.conf_data[-1]["num_timesteps"]+1)


	def plot_soc(self, name="PLAYER"):
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
		fig.add_trace(go.Scatter(x=self.xlims, y=self.data["Summary"]["OAT"][::2],
					mode='lines',
    				line_color='slateblue',
    				line_dash='dash',
					name="Outdoor Temp"),
					row=1, col=1, 
					secondary_y=False)
		fig.add_trace(go.Scatter(x=self.xlims, y=np.multiply(self.data["Summary"]["GHI"][::2],0.001),
					mode='lines',
    				line_color='darkorange',
    				line_dash="dot",
					name="GHI"),
					row=1, col=1, 
					secondary_y=True)

		fig.update_yaxes(title_text="<b>Temperature</b> [deg C]", row=1, col=1, secondary_y=False)
		fig.add_trace(go.Bar(x=self.xlims, y= np.multiply(-1,self.data[name]["hvac_cool_on_opt"]),  name="Cooling on/off", marker_color="dodgerblue"), row=1, col=1, secondary_y=True)
		fig.add_trace(go.Bar(x=self.xlims, y= self.data[name]["hvac_heat_on_opt"], name="Heat on/off", marker_color="firebrick"), row=1, col=1, secondary_y=True)

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

		for i in range(1,4):
			fig.add_trace(go.Bar(x=self.xlims, y=self.data[name]["correct_solve"], name="correct_solve"), row=1, col=i, secondary_y=True)

		return fig 

	def plot_community_peak(self):
		fig = go.Figure()
		for k,v in self.data.items():
			color = 'coral' if k == "PLAYER" else 'cadetblue'
			try:
				
				fig.add_trace(go.Scatter(x=self.xlims, y=v["p_grid_opt"], stackgroup='one', name=k, line_color=color))
			except:
				pass

		fig.update_layout(title_text="Community Demand")
		return fig 

	def main(self):
		# self.plot_soc().show()
		# self.plot_soc(name='Glenda-VUXFP').show()
		# self.plot_soc(name='Edward-HUHS3').show()
		self.plot_soc(name='Nadine-D73XI').show()
		self.plot_soc(name="Marquerite-EPELW").show()
		self.plot_community_peak().show()

if __name__=='__main__':
	p = Plotter()
	p.main()
