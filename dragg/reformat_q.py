import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_q_function(timestep, figure):
    th = data["theta"][timestep] # uses the final theta iteration

    s = np.linspace(min(data["state"]), max(data["state"]))
    a = np.linspace(min(data["action"]), max(data["action"]))
    sGrid, aGrid = np.meshgrid(s,a)

    # add surface of approx. Q function
    x = sGrid
    y = aGrid
    z = th[0] + th[1]*sGrid + th[2]*aGrid + th[3]*sGrid*aGrid + th[4]*sGrid**2 + th[5]*aGrid**2
    figure.add_trace(go.Surface(x=x, y=y, z=z))
    return figure

file = os.path.join("outputs","rl_agg","2015-01-01T00_2015-01-08T00-rl_agg_all-homes_20-horizon_8-q-results.json")
iterfile = os.path.join("outputs","rl_agg","2015-01-01T00_2015-01-08T00-rl_agg_all-homes_20-horizon_8-iter-results.json")


with open(file, 'r') as f:
    data = json.load(f)

with open(iterfile, 'r') as f:
    iterdata = json.load(f)

end = 48
th = data["theta"][end] # uses the final theta iteration

fig = make_subplots()
s = np.linspace(min(data["state"]), max(data["state"]))
a = np.linspace(min(data["action"]), max(data["action"]))
sGrid, aGrid = np.meshgrid(s,a)

# add surface of approx. Q function
x = sGrid
y = aGrid
z = th[0] + th[1]*sGrid + th[2]*aGrid + th[3]*sGrid*aGrid + th[4]*sGrid**2 + th[5]*aGrid**2
fig.add_trace(go.Surface(x=x, y=y, z=z))

# add scatter of recorded q-data
fig.add_trace(go.Scatter3d(x=data["state"][:end], y=data["action"][:end], z=data["q"][:end],mode="markers"))

fig.update_layout(scene = {
                        "xaxis_title":"State (P_agg)",
                        "yaxis_title":"Action (Reward Price ($/kWh))",
                        "zaxis_title":"Q-value"
})
fig.show()

fig2 = make_subplots()
for i in range(len(data["timestep"])-1):
    fig2.add_trace(go.Scatter(x=[1,2,3,4,5,6], y=data["theta"][i], mode='markers', marker={'color':'Blue'}))
    # fig2.add_trace(go.Scatter(x=[1,2,3,4,5,6], y=data["theta_k"][i], mode='markers', marker={'color':'Red'}))
fig2.show()

colors = px.colors.qualitative.Alphabet
fig3 = make_subplots()
for i in range(20):
    th = data["theta"][i]
    x = data["state"][i]
    u = np.linspace(-5,5)/100
    y = th[0] + th[1]*x + th[2]*u + th[3]*x*u + th[4]*x**2 + th[5]*u**2
    fig3.add_trace(go.Scatter(x=u, y=y, marker={'color':colors[i]}, name=f"Timestep {i} - Q function"))
    if iterdata[i]["is_greedy"]:
        width = 0
    else:
        width = 2
    fig3.add_trace(go.Scatter(x=[data["action"][i]], y=[data["q"][i]], mode="markers", marker={'color':colors[i], 'size':10, 'line':{'color':'Black', 'width':width}}, name=f"Timestep {i} - Selected u_k and observed q_k"))

fig3.update_layout(scene = {
                        "xaxis_title":"Action ($/kWh)",
                        "yaxis_title":"Q-value",
})

fig3.show()

fig4 = make_subplots()
fig4.add_trace(go.Scatter(x=data["timestep"], y=data["state"]))
fig4.show()
