# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import gc
import json
from sklearn.externals import joblib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("df10Kx20.csv")

# with open('dtypes.txt') as json_file:
    # dtypesR = json.load(json_file)
    
# df = pd.read_csv("code/FeatureDF.csv", dtype=dtypesR)
# def cleanKaggleTest(df):
    # train_df = df[df['TARGET'].notnull()]
    # test_df = df[df['TARGET'].isnull()]
    # del df
    # gc.collect()
    # return train_df, test_df

# df, testDF = cleanKaggleTest(df)
# del testDF
# gc.collect()

available_features = df.columns

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_features],
                value='EXT_SOURCE_1'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_features],
                value='EXT_SOURCE_2'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),

    # dcc.Slider(
        # id='year--slider',
        # min=df['Year'].min(),
        # max=df['Year'].max(),
        # value=df['Year'].max(),
        # step=None
    # )
])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('yaxis-type', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 yaxis_type):
    dff = df[[xaxis_column_name, yaxis_column_name, 'TARGET']].dropna()
    dff = dff[(dff[xaxis_column_name] > 0.1) & (dff[xaxis_column_name] < 0.2)]
    dff = dff[(dff[yaxis_column_name] > 0.7) & (dff[yaxis_column_name] < 0.9)]

    fig = px.scatter(x=dff[dff['TARGET'] == 1][xaxis_column_name],
                     y=dff[dff['TARGET'] == 1][yaxis_column_name],
                     color=dff[dff['TARGET'] == 1]['TARGET'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name, 
                     type='linear') 

    fig.update_yaxes(title=yaxis_column_name, 
                     type='linear') 

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)