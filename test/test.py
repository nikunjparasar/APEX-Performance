import fastf1 as ff1
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# Load data using fastf1
track = 'Silverstone'
session = ff1.get_session(2023, track, 'Q')
session.load()
lap = session.laps.pick_fastest()

# Prepare telemetry data
telemetry = lap.telemetry
x = telemetry['X']
y = telemetry['Y']
speed = telemetry['Speed']

# Create a DataFrame for easier handling
df = pd.DataFrame({'x': x, 'y': y, 'speed': speed})

# Create a Plotly figure using scatter with line mode
fig = px.scatter(df, x='x', y='y', color='speed', color_continuous_scale='Plasma',
                 labels={'speed': 'Speed (km/h)'})
fig.update_traces(mode='lines+markers', marker=dict(size=10), line=dict(width=2), connectgaps=False)
fig.update_layout(
    title=f'Fastest Lap Telemetry at {track}', 
    plot_bgcolor='#121212', 
    paper_bgcolor='#121212', 
    font=dict(color='white'),
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='track-map',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
