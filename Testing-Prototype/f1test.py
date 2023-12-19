import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Create traces for different parts of the car
car_body = go.Scatter(x=[0, 0, 1, 1, 0],
                      y=[0, 2, 2, 0, 0],
                      fill='toself',
                      fillcolor='white',
                      line=dict(color='black'),
                      name='car_body')

tire1 = go.Scatter(x=[0.2, 0.2, 0.8, 0.8, 0.2],
                   y=[0, -0.5, -0.5, 0, 0],
                   fill='toself',
                   fillcolor='black',
                   line=dict(color='black'),
                   name='tire1')

tire2 = go.Scatter(x=[0.2, 0.2, 0.8, 0.8, 0.2],
                   y=[2, 2.5, 2.5, 2, 2],
                   fill='toself',
                   fillcolor='black',
                   line=dict(color='black'),
                   name='tire2')

# Add the remaining parts of the car
tire3 = go.Scatter(x=[1, 1, 1.5, 1.5, 1],
                   y=[0.2, 0.2, 0.8, 0.8, 0.2],
                   fill='toself',
                   fillcolor='black',
                   line=dict(color='black'),
                   name='tire3')

tire4 = go.Scatter(x=[-0.5, -0.5, 0, 0, -0.5],
                   y=[0.2, 0.2, 0.8, 0.8, 0.2],
                   fill='toself',
                   fillcolor='black',
                   line=dict(color='black'),
                   name='tire4')

wing1 = go.Scatter(x=[0, 0.3, 0.3, 0.1, 0],
                   y=[1.5, 1.7, 1.8, 1.6, 1.5],
                   fill='toself',
                   fillcolor='red',
                   line=dict(color='black'),
                   name='wing1')

wing2 = go.Scatter(x=[1, 0.7, 0.7, 0.9, 1],
                   y=[1.5, 1.7, 1.8, 1.6, 1.5],
                   fill='toself',
                   fillcolor='red',
                   line=dict(color='black'),
                   name='wing2')

# Define the layout for the plot
layout = go.Layout(title='F1 Car',
                   xaxis=dict(range=[-0.5, 1.5], autorange=False),
                   yaxis=dict(range=[-0.5, 2.5], autorange=False),
                   height=600,
                   width=800,
                   showlegend=True)

# Add all the traces to the plot
fig = go.Figure(data=[car_body, tire1, tire2, tire3, tire4, wing1, wing2], layout=layout)

# Display the plot in a Dash app
app.layout = html.Div([
    dcc.Graph(id='f1-car', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    

