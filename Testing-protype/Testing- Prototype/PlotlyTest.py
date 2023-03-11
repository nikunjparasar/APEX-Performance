import numpy as np
from scipy.interpolate import splprep, splev
import dash
import plotly.graph_objects as go
from dash import dcc
from dash import html
from scipy.interpolate import interp1d


  #######################VEHICLE DYNAMICS SIMULATION##############################

def simulate_vehicle(track, initial_state, dt, N):
    """
    Simulates a vehicle along a given track.
    """
    x, y, psi, v = initial_state
    states = np.zeros((N, 4))
    states[0] = initial_state
    
    for i in range(1, N):
        # Calculate curvature of the track at the current position
        curvature = track.curvature(x, y)
        
        # Calculate lateral and longitudinal acceleration of the vehicle
        a_lat = v**2 * curvature
        a_long = 0
        
        # Calculate new state of the vehicle
        x += v * np.cos(psi) * dt
        y += v * np.sin(psi) * dt
        psi += v / track.L * np.tan(track.alpha(x, y, psi)) * dt
        v += (a_long - track.mu * track.g * np.cos(track.alpha(x, y, psi)) - a_lat * np.sin(track.alpha(x, y, psi))) * dt
        
        states[i] = np.array([x, y, psi, v])
        
    return states


def objective(params, track, initial_state, dt, N):
    """
    Calculates the objective function for a given set of parameters.
    """
    track.update_params(params)
    states = simulate_vehicle(track, initial_state, dt, N)
    
    # Calculate the total time to complete the track
    total_time = states[-1, 0] / states[-1, 3]
    
    # Calculate a penalty for deviating from the optimal racing line
    penalty = 0
    for i in range(N):
        penalty += track.dist_to_optimal(states[i, 0], states[i, 1])**2
    
    return total_time + track.lambda_ * penalty



#####################DASH APPLICATION##################

# Set up Dash app
app = dash.Dash(__name__)

# Define dropdown options
dropdown_options = [
    {'label': 'Silverstone', 'value': 'track_1'},
    {'label': 'Monza', 'value': 'track_2'}
]

# Define app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='track-dropdown',
        options=dropdown_options,
        value='track_1'
    ),
    dcc.Graph(id='track-plot')
])

# Define callback to update track plot based on dropdown selection
@app.callback(
    dash.dependencies.Output('track-plot', 'figure'),
    [dash.dependencies.Input('track-dropdown', 'value')]
)


def update_track_plot(track_name):
     # Load track data from CSV
    track_data_1 = np.genfromtxt('/Users/nikkparasar/Documents/Personal Projects/apexperformance/TrackModels/Silverstone.csv', delimiter=',')
    track_data_2 = np.genfromtxt('/Users/nikkparasar/Documents/Personal Projects/apexperformance/TrackModels/Monza.csv', delimiter=',')

    if track_name == 'track_1':
        track_data = track_data_1
    elif track_name == 'track_2':
        track_data = track_data_2


    # Extract data columns
    x_m = track_data[:, 0]
    y_m = track_data[:, 1]
    w_tr_right_m = track_data[:, 2]
    w_tr_left_m = track_data[:, 3]

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
    new_points = splev(np.linspace(0, 1, num=2000), tck)
    x_tr_left_smooth, y_tr_left_smooth = new_points


    # # optimal line
    # x_opt, y_opt = simulate_vehicle(objective, x_m, y_m, w_tr_right_m, w_tr_left_m, 200)


    # Create a Plotly figure with dark background
    fig = go.Figure()

    # Add track limits as lines with solid white color
    fig.add_trace(go.Scatter(x=x_tr_right_smooth, y=y_tr_right_smooth, line=dict(color='Teal', width=1), mode='lines', name='Right Track Limit'))
    fig.add_trace(go.Scatter(x=x_tr_left_smooth, y=y_tr_left_smooth, line=dict(color='Purple', width=1.5), mode='lines', name='Left Track Limit'))
    
    # Add the optimal line
    # fig.add_trace(go.Scatter(x=x_opt, y=y_opt, mode='lines', line=dict(color='red', width=1)))
    
    # Add center line as a line plot
    fig.add_trace(go.Scatter(x=x_m_smooth, y=y_m_smooth, line=dict(color='white', width=3), mode='lines', name='Center Line'))
    
    
    # Set figure layout and display plot
    fig.update_layout(template='plotly_dark', title='Track', xaxis=dict(visible=False), yaxis=dict(visible=False))
    # fig.update_layout(aspectmode='track_data')
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    


