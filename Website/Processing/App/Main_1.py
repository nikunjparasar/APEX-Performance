import numpy as np
from scipy.interpolate import splprep, splev
import dash
import plotly.graph_objects as go
from dash import dcc
from dash import html
import os
from scipy.optimize import minimize


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



# Define the dash application layout
app.layout = html.Div([
    dcc.Dropdown(
        id='track-dropdown',
        options=dropdown_options,
        value='track_01',
        style = {'backgroundColor': 'gray', 'width': '700px', 'font-family': 'Arial'},
        # option_style={'backgroundColor': 'gray'},

    ),
    dcc.Graph(
        id='track-plot', 
        style={'height': '700px', 'width': '700px'}
    )
    
])

# Define callback function to update track plot based on dropdown selection and physical value sliders
@app.callback(
    dash.dependencies.Output('track-plot', 'figure'),
    [dash.dependencies.Input('track-dropdown', 'value')]
)


def update_track_plot(track_name):
    
    
    ###########################################################################
    #                   LOADING CSV DATA FROM FILES                           #
    ###########################################################################
    
    cwd = os.getcwd()
    
    
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Austin.csv')
    austin_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'BrandsHatch.csv')
    bh_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Budapest.csv')
    budapest_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Catalunya.csv')
    cat_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Hockenheim.csv')
    hock_data = np.genfromtxt(file_path, delimiter=',')
    
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'IMS.csv')
    ims_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Melbourne.csv')
    mel_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'MexicoCity.csv')
    mc_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Montreal.csv')
    montreal_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Monza.csv')
    monza_data = np.genfromtxt(file_path, delimiter=',')
    
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'MoscowRaceway.csv')
    moscow_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Norisring.csv')
    noris_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Nuerburgring.csv')
    burg_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Oschersleben.csv')
    osch_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sakhir.csv')
    sakhir_data = np.genfromtxt(file_path, delimiter=',')
    
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'SaoPaulo.csv')
    sp_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sepang.csv')
    sepang_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Shanghai.csv')
    shanghai_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Silverstone.csv')
    silv_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sochi.csv')
    sochi_data = np.genfromtxt(file_path, delimiter=',')
    
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Spa.csv')
    spa_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Spielberg.csv')
    spiel_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Suzuka.csv')
    suzuka_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'YasMarina.csv')
    yasm_data = np.genfromtxt(file_path, delimiter=',')
    file_path = os.path.join(cwd, 'Processing/TrackModels', 'Zandvoort.csv')
    zandvoort_data = np.genfromtxt(file_path, delimiter=',')
    
    
    if track_name == 'track_01':
        track_data = austin_data
    elif track_name == 'track_02':
        track_data = bh_data
    elif track_name == 'track_03':
        track_data = budapest_data
    elif track_name == 'track_04':
        track_data = cat_data
    elif track_name == 'track_05':
        track_data = hock_data
        
    elif track_name == 'track_06':
        track_data = ims_data
    elif track_name == 'track_07':
        track_data = mel_data    
    elif track_name == 'track_08':
        track_data = mc_data
    elif track_name == 'track_09':
        track_data = montreal_data
    elif track_name == 'track_10':
        track_data = monza_data
        
    elif track_name == 'track_11':
        track_data = moscow_data
    elif track_name == 'track_12':
        track_data = noris_data
    elif track_name == 'track_13':
        track_data = burg_data    
    elif track_name == 'track_14':
        track_data = osch_data
    elif track_name == 'track_15':
        track_data = sakhir_data
        
    elif track_name == 'track_16':
        track_data = sp_data
    elif track_name == 'track_17':
        track_data = sepang_data
    elif track_name == 'track_18':
        track_data = shanghai_data
    elif track_name == 'track_19':
        track_data = silv_data    
    elif track_name == 'track_20':
        track_data = sochi_data
        
    elif track_name == 'track_21':
        track_data = spa_data
    elif track_name == 'track_22':
        track_data = spiel_data
    elif track_name == 'track_23':
        track_data = suzuka_data
    elif track_name == 'track_24':
        track_data = yasm_data   
    elif track_name == 'track_25':
        track_data = zandvoort_data   
        
    
    
    ###########################################################################
    #                       DATA POINT SMOOTHING                              #
    ###########################################################################
    
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


    ###########################################################################
    #                 PHYSICS CALCULATIONS                                    #
    ###########################################################################

    # The physical equations required for optimizing the lap time of a race car:
    
    
    '''

        Note:
        The following default values and parameters for a formula one car were found from:
        
        Limebeer, D. J., and G. Perantoni. “Optimal Control of a Formula One Car on a 
        Three-Dimensional Track—Part 2: Optimal Control.” Journal of Dynamic Systems, 
        Measurement, and Control, vol. 137, no. 5, 2015, 
        https://doi.org/10.1115/1.4029466. 
        
        I am using these parameters in my calcuations as well as they are representative of 
        the constraints that a high performance racecar undergoes during a qualifying lap.
        
        I have implemented these parameters as two classes in separate c++ algorithms for vehicle and tire modeling
        
        
            Vehicle Parameters 
            
        Symbol          Description                                     Default Value
        ------------------------------------------------------------------
        Pmax     |      Peak engine power                       |       560 kW
        M        |      Vehicle mass                            |       660 kg
        Ix       |      Moment of inertia about the x-axis      |       112.5 kg/m2
        Iy       |      Moment of inertia about the y-axis      |       450 kg/m2
        Iz       |      Moment of inertia about the z-axis      |       450 kg/m2
        W        |      Wheelbase                               |       3.4 m
        A        |      Distance of CoM from front axle         |       1.8 m
        B        |      Distance of CoM from rear axle          |       w-a
        H        |      Center of mass height                   |       0.3 m
        A        |      Frontal area                            |       1.5 m2    
        Droll    |      Roll moment distribution (@ front axle) |       0.5
        wf       |      Front wheel to car centerline distance  |       0.73 m
        wr       |      Rear wheel to car centerline distance   |       0.73 m
        R        |      Wheel radius                            |       0.33 m
        kd       |      Differential friction coefficient       |       10.47 Nm s/rad
        
        
                Tire Parameters
        Symbol          Description                                     Default Value
        ------------------------------------------------------------------
        Fz1      |      Reference load 1                        |       2000 N         
        Fz2      |      Reference load 2                        |       6000 N
        mux1     |      Peak longitudinal frict coef @ load 1   |       1.75
        mux2     |      Peak longitudinal frict coef @ load 2   |       1.40
        kappa1   |      Slip coef for the frict peak @ load 1   |       0.11
        kappa2   |      Slip coef for the frict peak @ load 2   |       0.10
        mu1      |      Peak lat friction coefficient @ load 1  |       1.80
        mu2      |      Peak lat friction coefficient @ load 2  |       1.45
        alpha1   |      Slip angle for the frict peak @ load 1  |       9 deg
        alpha2   |      Slip angle for the frict peak @ load 2  |       8 deg
        Qx       |      Longitudinal shape factor               |       1.9
        Qy       |      Lateral shape factor                    |       1.9

        
    '''
  

    #####################################################################
    ##                        PLOTTING DATA IN DASH                    ##
    ######################################################################


    # Create a Plotly figure with dark background
    fig = go.Figure()
    
    # Calculate optimal racing line
    # x_r, y_r = optimize_racing_line(x_m_smooth, y_m_smooth, x_tr_right_smooth, y_tr_right_smooth, x_tr_left_smooth, y_tr_left_smooth)
    # fig.add_trace(go.Scatter(x=x_r, y=y_r, line=dict(color='red', width=1), mode='lines', name='Racing Line (Gradient Descent)'))
    
    # Add track limits as lines with solid white color
    fig.add_trace(go.Scatter(x=x_tr_right_smooth, y=y_tr_right_smooth, line=dict(color='white', width=1), mode='lines', name='Right Track Limit'))
    fig.add_trace(go.Scatter(x=x_tr_left_smooth, y=y_tr_left_smooth, line=dict(color='white', width=1.5), mode='lines', name='Left Track Limit'))
    
  
    # Add center line as a line plot
    fig.add_trace(go.Scatter(x=x_m_smooth, y=y_m_smooth, line=dict(color='teal', width=0.5), mode='lines', name='Center Line'))
    
    
    # Set figure layout and display plot
    fig.update_layout(template='plotly_dark', title='Circuit Analysis', xaxis=dict(visible=False), yaxis=dict(visible=False))
    # fig.update_layout(aspectmode='track_data')
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    
    
    


