import fastf1 as ff1
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc

# Load session data
session = ff1.get_session(2021, 'Budapest', 'Q')
session.load()

fastest_lap = session.laps.pick_fastest()
car_data = fastest_lap.get_car_data().add_distance()
circuit_info = session.get_circuit_info()

# Convert data to DataFrame for easier manipulation
df = pd.DataFrame({'Distance': car_data['Distance'], 'Speed': car_data['Speed']})

# Create Plotly figure
fig = go.Figure()

# Add speed trace
# Instead of using fastf1.plotting for team color, you can set a color manually
# For example: team_color = 'blue' or any other color of your choice
team_color = 'teal'  # Replace with your preferred color
fig.add_trace(go.Scatter(x=df['Distance'], y=df['Speed'], mode='lines',
                         line=dict(color=team_color), name=fastest_lap['Driver']))

# Add vertical lines and corner numbers
v_min = df['Speed'].min()
v_max = df['Speed'].max()
for _, corner in circuit_info.corners.iterrows():
    fig.add_shape(type='line', x0=corner['Distance'], y0=v_min-20,
                  x1=corner['Distance'], y1=v_max+20, line=dict(color='grey', dash='dot'))
    fig.add_annotation(x=corner['Distance'], y=v_min-30, text=f"{corner['Number']}{corner['Letter']}",
                       showarrow=False, yshift=10)

# Update layout
fig.update_layout(
    title='Fastest Lap Telemetry',
    xaxis_title='Distance in m',
    yaxis_title='Speed in km/h',
    plot_bgcolor='black',
    yaxis=dict(range=[v_min - 40, v_max + 20])
)

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='v-plot',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
