import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev


# Load track data from CSV
data = np.genfromtxt('/Users/nikkparasar/Documents/Personal Projects/apexperformance/Processing/silverstone.csv', delimiter=',')

# Extract data columns
x_m = data[:, 0]
y_m = data[:, 1]
w_tr_right_m = data[:, 2]
w_tr_left_m = data[:, 3]

# Smooth the center line using cubic spline interpolation
tck, u = splprep([x_m, y_m], s=0)
new_points = splev(np.linspace(0, 1, num=1000), tck)
x_m_smooth, y_m_smooth = new_points

# Smooth the right track limit using cubic spline interpolation
dx_right = -w_tr_right_m*np.sin(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
dy_right = w_tr_right_m*np.cos(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
x_tr_right = x_m + dx_right
y_tr_right = y_m + dy_right

tck, u = splprep([x_tr_right, y_tr_right], s=0)
new_points = splev(np.linspace(0, 1, num=1000), tck)
x_tr_right_smooth, y_tr_right_smooth = new_points

# Smooth the left track limit using cubic spline interpolation
dx_left = w_tr_left_m*np.sin(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
dy_left = -w_tr_left_m*np.cos(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
x_tr_left = x_m + dx_left
y_tr_left = y_m + dy_left

tck, u = splprep([x_tr_left, y_tr_left], s=0)
new_points = splev(np.linspace(0, 1, num=1000), tck)
x_tr_left_smooth, y_tr_left_smooth = new_points

# Create a Plotly figure with dark background
fig = go.Figure()

# Add track limits as lines with solid white color
fig.add_trace(go.Scatter(x=x_tr_right_smooth, y=y_tr_right_smooth, line=dict(color='white', width=2), mode='lines', name='Track Limit'))
fig.add_trace(go.Scatter(x=x_tr_left_smooth, y=y_tr_left_smooth, line=dict(color='white', width=2), mode='lines', name='Track Limit'))

# Add center line as a line plot
# fig.add_trace(go.Scatter(x=x_m_smooth, y=y_m_smooth, line=dict(color='white', width=3), mode='lines', name='Center Line'))

# Set figure layout and display plot
fig.update_layout(template='plotly_dark', title='Track', xaxis=dict(visible=False), yaxis=dict(visible=False))
fig.show()