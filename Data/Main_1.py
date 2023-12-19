import numpy as np
import matplotlib as mpl
import fastf1 as ff1
import pandas as pd
import plotly.express as px

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import splprep, splev
import dash
import plotly.graph_objects as go
from dash import dcc
from dash import html
from Parse_RaceTracks import Parse_RaceTracks


# Set up Dash application
app = dash.Dash(__name__)

# Define the dropdown options for all 25 supported racetracks
dropdown_options = [
    {'label': 'Austin', 'value': 'track_01'},
    {'label': 'BrandsHatch', 'value': 'track_02'},
    {'label': 'Budapest', 'value': 'track_03'},
    {'label': 'Catalunya', 'value': 'track_04'},
    {'label': 'Hockenheim', 'value': 'track_05'},
    {'label': 'IMS', 'value': 'track_06'},
    {'label': 'Melbourne', 'value': 'track_07'},
    {'label': 'Mexico City', 'value': 'track_08'},
    {'label': 'Montreal', 'value': 'track_09'},
    {'label': 'Monza', 'value': 'track_10'},
    {'label': 'Moscow Raceway', 'value': 'track_11'},
    {'label': 'Norisring', 'value': 'track_12'},
    {'label': 'Neurburgring', 'value': 'track_13'},
    {'label': 'Oschersleben', 'value': 'track_14'},
    {'label': 'Sakhir', 'value': 'track_15'},
    {'label': 'Sao Paulo', 'value': 'track_16'},
    {'label': 'Sepang', 'value': 'track_17'},
    {'label': 'Shanghai', 'value': 'track_18'},
    {'label': 'Silverstone', 'value': 'track_19'},
    {'label': 'Sochi', 'value': 'track_20'},
    {'label': 'Spa', 'value': 'track_21'},
    {'label': 'Spielberg', 'value': 'track_22'},
    {'label': 'Suzuka', 'value': 'track_23'},
    {'label': 'Yas Marina', 'value': 'track_24'},
    {'label': 'Zandvoort', 'value': 'track_25'},
]

value_to_name_mapping = {option['value']: option['label'] for option in dropdown_options}




# Define the dash application layout
app.layout = html.Div([
    dcc.Dropdown(
        id='track-dropdown',
        options=dropdown_options,
        value='track_01',
        style={'backgroundColor': 'lime', 'width': '100%', 'font-family': 'Arial'},
    ),
    html.Div([
        dcc.Graph(
            id='track-plot',
            style={'height': '700px', 'width': '50%'}
        ),
        dcc.Graph(
            id='speed-plot',
            style={'height': '700px', 'width': '50%'}
        )
    ], style={'display': 'flex', 'backgroundColor': 'black'}),
], style={'backgroundColor': 'black'})

@app.callback(
    [dash.dependencies.Output('track-plot', 'figure'),
     dash.dependencies.Output('speed-plot', 'figure')],
    [dash.dependencies.Input('track-dropdown', 'value')]
)

def update_track_plot(track_name):
    r_fig = update_racing_line(track_name)
    s_fig = update_speed_plot(track_name)
    return r_fig, s_fig

def update_speed_plot(track_name):
    track = value_to_name_mapping[track_name]
    session = ff1.get_session(2023, track, 'Q')
    session.load()
    lap = session.laps.pick_fastest()
    telemetry = lap.telemetry
    x = telemetry['X']
    y = telemetry['Y']
    speed = telemetry['Speed']
    
    df = pd.DataFrame({'x': x, 'y': y, 'speed': speed})

    fig = px.scatter(df, x='x', y='y', color='speed', color_continuous_scale='Plasma',
                 labels={'speed': 'Speed (km/h)'})
    fig.update_traces(mode='lines+markers', marker=dict(size=7), line=dict(width=2), connectgaps=False)
    fig.update_layout(
        title=f'Fastest Lap Telemetry at {track}', 
        plot_bgcolor='#121212', 
        paper_bgcolor='#121212', 
        font=dict(color='white'),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)    
    )

    return fig

def update_racing_line(track_name):
        
    track_data, line_data = Parse_RaceTracks.parse(track_name)
    
    # Here I am smoothing the data points to resolve any rough edes and sharp verticies 
    # because the data points I am currently using are not extremely precise, so some curvature
    # estimation is done through cubic spline interpolation, but I think this should still provide 
    # accurate enough track limits for now. Later I actually plan to use GPX files for coordinate 
    # specific track imaging, along with 3d spline curves to model the tracks with elevation and 
    # slope changes as well. 
    
    # Extract data columns
    x_m = track_data[:, 0]
    y_m = track_data[:, 1]
    w_tr_right_m = track_data[:, 2]
    w_tr_left_m = track_data[:, 3]

    # Smooth the center line using cubic spline interpolation
    tck, u = splprep([x_m, y_m], s=0)
    new_points = splev(np.linspace(0, 1, num=1000), tck)
    x_m_smooth, y_m_smooth = new_points

    # Smooth the right limit using cubic spline interpolation
    dx_right = -w_tr_right_m*np.sin(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
    dy_right = w_tr_right_m*np.cos(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
    x_tr_right = x_m + dx_right
    y_tr_right = y_m + dy_right

    tck, u = splprep([x_tr_right, y_tr_right], s=0)
    new_points = splev(np.linspace(0, 1, num=1000), tck)
    x_tr_right_smooth, y_tr_right_smooth = new_points

    # Smooth the left limit using cubic spline interpolation
    dx_left = w_tr_left_m*np.sin(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
    dy_left = -w_tr_left_m*np.cos(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
    x_tr_left = x_m + dx_left
    y_tr_left = y_m + dy_left

    tck, u = splprep([x_tr_left, y_tr_left], s=0)
    new_points = splev(np.linspace(0, 1, num=2000), tck)
    x_tr_left_smooth, y_tr_left_smooth = new_points
    
    
    # extract data for racing lines
    x_r = line_data[:, 0]
    y_r = line_data[:, 1]
    

    # Create a Plotly figure with dark background
    fig = go.Figure()
    
    # Calculate optimal racing line
    # x_r, y_r = optimize_racing_line(x_m_smooth, y_m_smooth, x_tr_right_smooth, y_tr_right_smooth, x_tr_left_smooth, y_tr_left_smooth)
    # fig.add_trace(go.Scatter(x=x_r, y=y_r, line=dict(color='red', width=1), mode='lines', name='Racing Line (Gradient Descent)'))
    
    # Add track limits as lines with solid white color
    fig.add_trace(go.Scatter(x=x_tr_right_smooth, y=y_tr_right_smooth, line=dict(color='white', width=3), mode='lines', name='Right Track Limit'))
    fig.add_trace(go.Scatter(x=x_tr_left_smooth, y=y_tr_left_smooth, line=dict(color='white', width=3), mode='lines', name='Left Track Limit'))
    
  
    # Add center line as a line plot
    fig.add_trace(go.Scatter(x=x_m_smooth, y=y_m_smooth, line=dict(color='red', width=0.5), mode='lines', name='Center Line'))
    
    # add racing line as a line plot
    fig.add_trace(go.Scatter(x=x_r, y=y_r, line=dict(color='#00FF00', width=4), mode='lines', name='Optimal Racing Line'))

    
    # Set layout and display plot
    
    fig.update_layout(template='plotly_dark', title='APEX Race Line Analysis', xaxis=dict(visible=False), yaxis=dict(visible=False))
    # fig.update_layout(aspectmode='track_data')
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    
    
    


