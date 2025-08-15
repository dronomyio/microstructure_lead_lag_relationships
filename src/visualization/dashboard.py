import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
from datetime import datetime

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Lead-Lag Analysis Dashboard'),
    
    html.Div([
        html.Div([
            html.H3('Cross-Correlation Heatmap'),
            dcc.Graph(id='correlation-heatmap'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3('Lead-Lag Distribution'),
            dcc.Graph(id='lag-distribution'),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        html.H3('Time Series with Detected Lead-Lag'),
        dcc.Graph(id='timeseries-plot'),
    ]),
    
    html.Div([
        html.H3('Information Ratio by Exchange Pair'),
        dcc.Graph(id='info-ratio-plot'),
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every second
        n_intervals=0
    )
])

@app.callback(
    [Output('correlation-heatmap', 'figure'),
     Output('lag-distribution', 'figure'),
     Output('timeseries-plot', 'figure'),
     Output('info-ratio-plot', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    # Load latest results
    try:
        with open('/app/output/lead_lag_results.json', 'r') as f:
            results = json.load(f)
    except:
        # Return empty figures if no data
        return {}, {}, {}, {}
    
    df = pd.DataFrame(results)
    
    # Correlation heatmap
    pivot_corr = df.pivot_table(
        values='correlation',
        index='exchange1',
        columns='exchange2',
        aggfunc='mean'
    )
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=pivot_corr.values,
        x=pivot_corr.columns,
        y=pivot_corr.index,
        colorscale='RdBu',
        zmid=0
    ))
    heatmap_fig.update_layout(
        title='Average Cross-Correlation by Exchange Pair',
        xaxis_title='Exchange 2',
        yaxis_title='Exchange 1'
    )
    
    # Lag distribution
    lag_fig = go.Figure(data=[
        go.Histogram(
            x=df['lag_nanoseconds'],
            nbinsx=50,
            name='Lead-Lag Distribution'
        )
    ])
    lag_fig.update_layout(
        title='Distribution of Optimal Lags (nanoseconds)',
        xaxis_title='Lag (ns)',
        yaxis_title='Frequency'
    )
    
    # Time series example (mock data for visualization)
    t = np.arange(1000)
    lag_samples = int(df['lag_nanoseconds'].mean() / 1000)  # Convert to sample units
    
    signal1 = np.sin(2 * np.pi * 0.01 * t) + 0.5 * np.random.randn(len(t))
    signal2 = np.roll(signal1, lag_samples) + 0.1 * np.random.randn(len(t))
    
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(
        x=t, y=signal1,
        mode='lines',
        name='Exchange 1',
        line=dict(color='blue', width=1)
    ))
    ts_fig.add_trace(go.Scatter(
        x=t, y=signal2,
        mode='lines',
        name='Exchange 2',
        line=dict(color='red', width=1)
    ))
    ts_fig.update_layout(
        title=f'Time Series with Detected Lag: {lag_samples} samples',
        xaxis_title='Time (samples)',
        yaxis_title='Price'
    )
    
    # Information ratio plot
    info_ratio_fig = go.Figure(data=[
        go.Bar(
            x=df.groupby('exchange1')['information_ratio'].mean().index,
            y=df.groupby('exchange1')['information_ratio'].mean().values,
            name='Information Ratio'
        )
    ])
    info_ratio_fig.update_layout(
        title='Average Information Ratio by Exchange',
        xaxis_title='Exchange',
        yaxis_title='Information Ratio'
    )
    
    return heatmap_fig, lag_fig, ts_fig, info_ratio_fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8051, debug=True)
